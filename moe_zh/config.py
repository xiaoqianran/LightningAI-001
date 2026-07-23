from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    n_expert: int = 4
    top_k: int = 1
    expert_ffn_mult: int = 2
    vocab_size: int = 8000
    block_size: int = 256
    dropout: float = 0.0


@dataclass
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    balance_loss_coef: float = 0.01
    log_every: int = 20
    eval_every: int = 200
    ckpt_every: int = 200
    seed: int = 42
    num_workers: int = 0


@dataclass
class DataConfig:
    max_bytes: int = 30_000_000
    train_ratio: float = 0.98
    seed: int = 42


@dataclass
class PathsConfig:
    data_dir: str = "data/d0"
    artifact_dir: str = "artifacts"
    tokenizer_prefix: str = "spm_zh_8k"


@dataclass
class GenerateConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50
    prompts: list[str] = field(default_factory=lambda: ["今天天气", "人工智能是"])


@dataclass
class SmokeConfig:
    name: str = "smoke"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)

    def to_dict(self) -> dict[str, Any]:
        def _conv(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {f.name: _conv(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, list):
                return list(obj)
            return obj

        return _conv(self)


def _merge_dataclass(cls: type, data: dict | None) -> Any:
    if not data:
        return cls()
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_config(path: str | Path) -> SmokeConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SmokeConfig(
        name=raw.get("name", "smoke"),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        train=_merge_dataclass(TrainConfig, raw.get("train")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        paths=_merge_dataclass(PathsConfig, raw.get("paths")),
        generate=_merge_dataclass(GenerateConfig, raw.get("generate")),
    )
