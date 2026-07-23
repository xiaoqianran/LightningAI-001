# 002 — 最小简体中文 MoE

S1 smoke：小 MoE（4 层 / 256 维 / 4 expert top-1）+ 中文语料 + 可 resume。

## 用法

```bash
# 根目录
python main.py 002 prepare --force-sample
python main.py 002 train --max-steps 200
python main.py 002 generate
python main.py 002 job              # 远程 Lightning Job

# 或
cd 002
python run.py prepare --force-sample
python run.py train --max-steps 50
```

## 依赖

```bash
cd 002
python -m venv .venv && .venv/bin/pip install -e .
# 或 pip install -r 见 pyproject.toml
```

## 子命令

| 命令 | 作用 |
|------|------|
| `prepare` | 公开/样例数据 + SentencePiece + bin |
| `train` | 训练（`--resume auto` 默认） |
| `generate` | 从 `artifacts/checkpoints/last.pt` 生成 |
| `job` | 提交远程 Job（默认 CPU；可用 `LIGHTNING_MACHINE`） |

## 结构

```text
002/
  run.py
  configs/smoke.yaml
  data/sample/zh_smoke.txt
  moe_zh/                 # 模型与训练包
  artifacts/              # 本地产物（gitignore）
```

设计说明见仓库 `docs/002-s1-smoke-design.md`。
