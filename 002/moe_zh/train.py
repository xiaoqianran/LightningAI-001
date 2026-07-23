#!/usr/bin/env python3
"""Train minimal Chinese MoE (S1 smoke) with checkpoint resume."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler

from moe_zh.config import ModelConfig, SmokeConfig, load_config
from moe_zh.data import TokenBinDataset
from moe_zh.model import MoELanguageModel

_REPO = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, cfg_train: Any) -> float:
    if step < cfg_train.warmup_steps:
        return cfg_train.lr * (step + 1) / max(cfg_train.warmup_steps, 1)
    if step >= cfg_train.max_steps:
        return cfg_train.lr * cfg_train.min_lr_ratio
    progress = (step - cfg_train.warmup_steps) / max(
        1, cfg_train.max_steps - cfg_train.warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg_train.lr * cfg_train.min_lr_ratio
    return min_lr + coeff * (cfg_train.lr - min_lr)


def resolve_artifact_dir(cfg: SmokeConfig) -> Path:
    env = os.environ.get("MOE_ARTIFACT_DIR")
    if env:
        return Path(env)
    return _REPO / cfg.paths.artifact_dir


def resolve_data_dir(cfg: SmokeConfig) -> Path:
    p = Path(cfg.paths.data_dir)
    if not p.is_absolute():
        p = _REPO / p
    return p


def build_model(cfg: SmokeConfig, data_dir: Path, device: torch.device) -> MoELanguageModel:
    vs_file = data_dir / "vocab_size.txt"
    if vs_file.exists():
        vs = int(vs_file.read_text(encoding="utf-8").strip())
        cfg.model.vocab_size = vs
    model = MoELanguageModel(cfg.model).to(device)
    return model


@torch.no_grad()
def evaluate(
    model: MoELanguageModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, y)
        assert loss is not None
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(n, 1)


def save_ckpt(
    path: Path,
    model: MoELanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: SmokeConfig,
    metrics: dict[str, Any],
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.to_dict(),
        "model_config": model.config_dict(),
        "metrics": metrics,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, path)


def load_ckpt(
    path: Path,
    model: MoELanguageModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0))


def train(cfg: SmokeConfig, resume: str = "auto", max_steps_override: int | None = None) -> int:
    if max_steps_override is not None:
        cfg.train.max_steps = max_steps_override

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.train.seed)
    data_dir = resolve_data_dir(cfg)
    artifact_dir = resolve_artifact_dir(cfg)
    ckpt_dir = artifact_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    if not train_bin.exists():
        print(f"missing {train_bin}; run: python -m moe_zh.prepare_d0", file=sys.stderr)
        return 2

    train_ds = TokenBinDataset(train_bin, cfg.model.block_size)
    val_ds = TokenBinDataset(val_bin, cfg.model.block_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        sampler=RandomSampler(train_ds, replacement=True, num_samples=10**9),
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(cfg, data_dir, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    start_step = 0
    last_path = ckpt_dir / "last.pt"
    if resume == "auto" and last_path.exists():
        start_step = load_ckpt(last_path, model, optimizer, device, scaler)
        print(f"resumed from {last_path} at step={start_step}")
    elif resume not in ("auto", "none", "") and Path(resume).exists():
        start_step = load_ckpt(Path(resume), model, optimizer, device, scaler)
        print(f"resumed from {resume} at step={start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} max_steps={cfg.train.max_steps} start={start_step}")

    metrics: dict[str, Any] = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "ce_loss": [],
        "balance_loss": [],
    }
    expert_totals = torch.zeros(cfg.model.n_expert, dtype=torch.long)

    model.train()
    it = iter(train_loader)
    t0 = time.time()
    running = 0.0
    running_n = 0

    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step, cfg.train)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        ce_accum = 0.0
        bal_accum = 0.0
        for _ in range(cfg.train.grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                _, ce_loss, aux = model(x, y)
                assert ce_loss is not None
                bal = aux["balance_raw"]
                loss = (ce_loss + cfg.train.balance_loss_coef * bal) / cfg.train.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += float(loss.item()) * cfg.train.grad_accum
            ce_accum += float(aux["ce_loss"].item())
            bal_accum += float(aux["balance_loss"].item())
            expert_totals += aux["expert_counts"].cpu()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        running += loss_accum
        running_n += 1

        if (step + 1) % cfg.train.log_every == 0 or step == start_step:
            avg = running / max(running_n, 1)
            dt = time.time() - t0
            print(
                f"step {step+1}/{cfg.train.max_steps} loss={avg:.4f} "
                f"ce={ce_accum/cfg.train.grad_accum:.4f} bal={bal_accum/cfg.train.grad_accum:.4f} "
                f"lr={lr:.2e} elapsed={dt:.1f}s"
            )
            metrics["steps"].append(step + 1)
            metrics["train_loss"].append(avg)
            metrics["ce_loss"].append(ce_accum / cfg.train.grad_accum)
            metrics["balance_loss"].append(bal_accum / cfg.train.grad_accum)
            metrics["lr"].append(lr)
            running = 0.0
            running_n = 0

        do_eval = (step + 1) % cfg.train.eval_every == 0 or (step + 1) == cfg.train.max_steps
        if do_eval:
            vloss = evaluate(model, val_loader, device)
            metrics["val_loss"].append({"step": step + 1, "loss": vloss})
            print(f"  val_loss={vloss:.4f}")

        do_ckpt = (step + 1) % cfg.train.ckpt_every == 0 or (step + 1) == cfg.train.max_steps
        if do_ckpt:
            snap = {
                "last_train_loss": metrics["train_loss"][-1] if metrics["train_loss"] else None,
                "val": metrics["val_loss"][-1] if metrics["val_loss"] else None,
            }
            save_ckpt(last_path, model, optimizer, step + 1, cfg, snap, scaler)
            step_path = ckpt_dir / f"step_{step+1:06d}.pt"
            save_ckpt(step_path, model, optimizer, step + 1, cfg, snap, scaler)
            print(f"  saved {last_path.name} and {step_path.name}")

    # write metrics + router stats
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    total = int(expert_totals.sum().item()) or 1
    fracs = (expert_totals.float() / total).tolist()
    router = {
        "expert_counts": expert_totals.tolist(),
        "expert_fractions": fracs,
        "n_expert": cfg.model.n_expert,
    }
    (artifact_dir / "router_stats.json").write_text(
        json.dumps(router, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # copy tokenizer into artifacts
    tok_src = data_dir / "tokenizer"
    tok_dst = artifact_dir / "tokenizer"
    if tok_src.exists():
        tok_dst.mkdir(parents=True, exist_ok=True)
        for f in tok_src.glob("*"):
            target = tok_dst / f.name
            if not target.exists():
                target.write_bytes(f.read_bytes())

    meta = {
        "device": str(device),
        "params": n_params,
        "max_steps": cfg.train.max_steps,
        "start_step": start_step,
        "final_step": cfg.train.max_steps,
        "elapsed_sec": time.time() - t0,
        "cuda": torch.cuda.is_available(),
    }
    (artifact_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"done. metrics -> {metrics_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(_REPO / "configs" / "smoke.yaml"))
    p.add_argument("--resume", default="auto", help="auto | none | path/to.pt")
    p.add_argument("--max-steps", type=int, default=None, help="override max_steps (CPU tests)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    return train(cfg, resume=args.resume, max_steps_override=args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
