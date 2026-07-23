#!/usr/bin/env python3
"""Generate Simplified Chinese samples from a smoke checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from moe_zh.config import ModelConfig, load_config
from moe_zh.model import MoELanguageModel
from moe_zh.tokenizer_util import Tokenizer

_REPO = Path(__file__).resolve().parents[1]


def load_model(ckpt_path: Path, device: torch.device) -> tuple[MoELanguageModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ckpt.get("model_config") or ckpt.get("config", {}).get("model") or {}
    cfg = ModelConfig(**{k: mc[k] for k in ModelConfig.__dataclass_fields__ if k in mc})
    model = MoELanguageModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(_REPO / "configs" / "smoke.yaml"))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    artifact = _REPO / cfg.paths.artifact_dir
    ckpt_path = Path(args.ckpt) if args.ckpt else artifact / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    tok_path = (
        Path(args.tokenizer)
        if args.tokenizer
        else artifact / "tokenizer" / f"{cfg.paths.tokenizer_prefix}.model"
    )
    if not tok_path.exists():
        # fallback data dir
        alt = _REPO / cfg.paths.data_dir / "tokenizer" / f"{cfg.paths.tokenizer_prefix}.model"
        tok_path = alt
    if not tok_path.exists():
        print(f"tokenizer not found: {tok_path}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(ckpt_path, device)
    tok = Tokenizer(tok_path)

    lines = [f"# generate from {ckpt_path} step={ckpt.get('step')}"]
    for prompt in cfg.generate.prompts:
        ids = tok.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            x,
            max_new_tokens=cfg.generate.max_new_tokens,
            temperature=cfg.generate.temperature,
            top_k=cfg.generate.top_k,
        )
        text = tok.decode(out[0].tolist())
        lines.append(f"## prompt: {prompt}")
        lines.append(text)
        lines.append("")
        print(f">>> {prompt}")
        print(text)
        print()

    out_path = Path(args.out) if args.out else artifact / "generations_smoke.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
