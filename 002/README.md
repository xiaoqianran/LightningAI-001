# 002 — 最小简体中文 MoE

| 阶段 | config | 说明 |
|------|--------|------|
| **S1 smoke** | `smoke` | 4 层 / 256 / 4 expert top-1，样例或小数据，管线验证 |
| **S2 mini** | `mini` | 6 层 / 384 / 8 expert top-2，公开中文 ~250MB，5000 step |

## 用法

```bash
# S1
python main.py 002 prepare --config smoke --force-sample
python main.py 002 train --config smoke --max-steps 200
python main.py 002 job --config smoke --machine CPU

# S2 mini（默认远程 T4 ≈ $0.19/h，interruptible）
python main.py 002 prepare --config mini
python main.py 002 train --config mini
python main.py 002 job --config mini --machine T4 --interruptible
```

## 子命令

| 命令 | 作用 |
|------|------|
| `prepare` | 数据 + SentencePiece + bin（`--config smoke\|mini`） |
| `train` | 训练（`--resume auto`） |
| `generate` | 从 last.pt 生成 |
| `job` | 远程 Lightning Job |

## GPU 选型（S2 默认）

见 `docs/002-gpu-and-s2.md`：标价 T4 **~$0.19/h**（最便宜且够用），A100 ~$2.19，H100 ~$4.50（超 $3 预算不用）。

## 结构

```text
002/
  run.py
  configs/smoke.yaml
  configs/mini.yaml
  data/sample/
  moe_zh/
  artifacts/          # gitignore
```
