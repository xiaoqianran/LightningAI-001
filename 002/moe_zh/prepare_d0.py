#!/usr/bin/env python3
"""Prepare D0 Simplified-Chinese corpus + SentencePiece + token bins."""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

from moe_zh.config import load_config
from moe_zh.data import write_token_bin
from moe_zh.tokenizer_util import Tokenizer, train_sentencepiece

_REPO = Path(__file__).resolve().parents[1]
_SAMPLE = _REPO / "data" / "sample" / "zh_smoke.txt"


def _clean_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    line = re.sub(r"https?://\S+", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) < 10 or len(line) > 2000:
        return None
    # prefer lines with CJK
    if not re.search(r"[\u4e00-\u9fff]", line):
        return None
    return line


def _load_sample(max_bytes: int) -> list[str]:
    if not _SAMPLE.exists():
        raise FileNotFoundError(f"missing fallback sample: {_SAMPLE}")
    text = _SAMPLE.read_text(encoding="utf-8")
    lines = []
    # repeat sample content to reach a usable size for smoke
    base = [ln for ln in (_clean_line(x) for x in text.splitlines()) if ln]
    if not base:
        raise RuntimeError("sample file produced no valid lines")
    total = 0
    i = 0
    while total < max_bytes and i < 500_000:
        ln = base[i % len(base)]
        # slight variation so SP sees more patterns when repeating
        if i // len(base) > 0 and i % 7 == 0:
            ln = ln + "。"
        lines.append(ln)
        total += len(ln.encode("utf-8"))
        i += 1
    return lines


def _load_wikipedia_zh(max_bytes: int) -> list[str] | None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("datasets not installed; skip wikipedia download", file=sys.stderr)
        return None
    try:
        # streaming to avoid huge download
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.zh",
            split="train",
            streaming=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"wikipedia load failed: {exc}", file=sys.stderr)
        return None

    lines: list[str] = []
    total = 0
    for row in ds:
        text = row.get("text") or ""
        for raw in text.splitlines():
            ln = _clean_line(raw)
            if not ln:
                continue
            lines.append(ln)
            total += len(ln.encode("utf-8"))
            if total >= max_bytes:
                return lines
        if total >= max_bytes:
            break
    return lines if lines else None


def prepare(
    out_dir: Path,
    max_bytes: int,
    vocab_size: int,
    train_ratio: float,
    seed: int,
    tokenizer_prefix: str,
    force_sample: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] | None = None
    source = "sample"
    if not force_sample:
        print(f"fetching public zh data (max_bytes={max_bytes})...")
        lines = _load_wikipedia_zh(max_bytes)
        if lines:
            source = "wikimedia/wikipedia:20231101.zh"
    if not lines:
        print("using local sample fallback (expanded)")
        # for sample, don't need full 30MB unless requested
        mb = min(max_bytes, 2_000_000) if force_sample else min(max_bytes, 5_000_000)
        lines = _load_sample(mb)
        source = "data/sample/zh_smoke.txt"

    rng = random.Random(seed)
    rng.shuffle(lines)
    n_train = max(1, int(len(lines) * train_ratio))
    train_lines = lines[:n_train]
    val_lines = lines[n_train:] or lines[: max(1, len(lines) // 50)]

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    train_txt.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    meta = out_dir / "source.txt"
    meta.write_text(
        f"source={source}\nlines={len(lines)}\ntrain={len(train_lines)}\nval={len(val_lines)}\n",
        encoding="utf-8",
    )
    print(f"wrote {train_txt} ({len(train_lines)} lines), {val_txt} ({len(val_lines)} lines)")

    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    prefix = tok_dir / tokenizer_prefix
    # hard_vocab_limit=False allows SP to shrink if corpus is tiny
    vs = vocab_size
    try:
        model_path = train_sentencepiece(train_txt, prefix, vocab_size=vs)
    except Exception as exc:  # noqa: BLE001
        vs = min(vocab_size, 512)
        print(f"SP train failed ({exc}); retry vocab_size={vs}", file=sys.stderr)
        model_path = train_sentencepiece(train_txt, prefix, vocab_size=vs)

    tok = Tokenizer(model_path)
    print(f"tokenizer vocab_size={tok.vocab_size} path={model_path}")

    def encode_file(txt_path: Path) -> list[int]:
        ids: list[int] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids.extend(tok.encode(line, add_bos=True, add_eos=True))
        return ids

    train_ids = encode_file(train_txt)
    val_ids = encode_file(val_txt)
    # ensure minimum length for block training
    min_len = 512
    if len(train_ids) < min_len:
        train_ids = (train_ids * ((min_len // max(len(train_ids), 1)) + 2))[: min_len * 4]
    if len(val_ids) < min_len:
        val_ids = (val_ids * ((min_len // max(len(val_ids), 1)) + 2))[:min_len]

    n_tr = write_token_bin(train_ids, out_dir / "train.bin")
    n_va = write_token_bin(val_ids, out_dir / "val.bin")
    print(f"bins: train.bin={n_tr} tokens, val.bin={n_va} tokens")
    (out_dir / "vocab_size.txt").write_text(str(tok.vocab_size), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare D0 Chinese data for MoE smoke")
    p.add_argument("--config", default=str(_REPO / "configs" / "smoke.yaml"))
    p.add_argument("--out", default=None, help="output dir (default: config paths.data_dir)")
    p.add_argument("--max-bytes", type=int, default=None)
    p.add_argument("--force-sample", action="store_true", help="skip network, use sample only")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out) if args.out else _REPO / cfg.paths.data_dir
    max_bytes = args.max_bytes if args.max_bytes is not None else cfg.data.max_bytes
    prepare(
        out_dir=out,
        max_bytes=max_bytes,
        vocab_size=cfg.model.vocab_size,
        train_ratio=cfg.data.train_ratio,
        seed=cfg.data.seed,
        tokenizer_prefix=cfg.paths.tokenizer_prefix,
        force_sample=args.force_sample,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
