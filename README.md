# LightningAI-Lab

Lightning AI 实验台：编号目录做最小可跑 Job / 训练冒烟（风格对齐 [cuda-lab](https://github.com/xiaoqianran/cuda-lab)）。

## 结构

```text
main.py          # 入口，调度到 001 / 002 / ...
001/             # Hello Job（CPU Docker 冒烟）
002/             # 最小简体中文 MoE（prepare / train / generate / job）
docs/            # 计划与设计说明
```

## 环境

- Python 3.11+
- [Lightning AI](https://lightning.ai) 账号：`LIGHTNING_USER_ID` + `LIGHTNING_API_KEY` 或 `lightning login`
- 001：`pip install lightning-sdk`
- 002：见 `002/pyproject.toml`（torch / sentencepiece 等）

## 用法

```bash
# 001 Hello Job（提交到 Lightning 并等待）
python main.py 001 run

# 002 S1 smoke
python main.py 002 prepare --config smoke --force-sample
python main.py 002 train --config smoke --max-steps 200

# 002 S2 mini（远程默认 T4，公开数据）
python main.py 002 job --config mini --machine T4 --interruptible
```

也可进入实验目录：

```bash
cd 001 && python run.py run
cd 002 && python run.py train --config mini --max-steps 100
```

## 约定

- 每个实验自包含：`run.py` + `README.md` + 可选 `pyproject.toml`
- 数据 / 权重默认不入库；生成于 `data/d0|d1/`、`artifacts/`
- 远程 Job 默认 teamspace：`seachenxyt/seachenxyt`（可用环境变量覆盖）
- GPU：同卡选更便宜路径；预算 &lt; $3/h；S2 默认 **T4**（见 `docs/002-gpu-and-s2.md`）

## 仓库

- 远程：https://github.com/xiaoqianran/LightningAI-Lab  
- 前身：`LightningAI-001`（已重命名）
