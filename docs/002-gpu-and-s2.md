# GPU 选型与 S2 Mini

更新：2026-07-24（账号 `seachenxyt` / teamspace `seachenxyt`）

## 1. 平台标价（Lightning 官方 pricing 页）

来源：[lightning.ai/pricing](https://lightning.ai/pricing/)（按 **GPU·小时**，会变动）

| 机型 | VRAM | 标价 $/h（约） | 相对预算 (&lt;$3/h) |
|------|------|----------------|-------------------|
| **T4** | 16 GB | **$0.19** | ✅ 最便宜 GPU |
| **L4** | 24 GB | **$0.48** | ✅ |
| L40S | 48 GB | $2.14 | ✅ |
| **A100 40GB** | 40 GB | **$2.19** | ✅ 在允许高端池内 |
| A100 80GB | 80 GB | $2.71 | ✅ |
| **RTXP 6000** | 96 GB | **$2.89** | ✅ 贴近上限 |
| H100 80GB | 80 GB | **$4.50** | ❌ **超过 $3/h** |

多卡 `_X_N` 通常按卡数倍增，S2 默认 **单卡**。

本仓库 **一律 on-demand**，不以 spot 报价为准。

## 2. 本账号实测可提交性

对 `seachenxyt/seachenxyt` 用 SDK `Job.run` 探测（枚举 `Machine.*` + 可选 `cloud`）：

| 结果 | 说明 |
|------|------|
| **CPU / T4** | 曾成功创建 Job（含 default / AWS / GCP 组合） |
| 部分时刻全挂 | 报 `accelerator X not found for this AWS cluster` — 与默认 cloud 绑定有关，换 cloud 或重试可恢复 |
| 定价 API | 无公开稳定 `/pricing` REST；以官网标价 + Job 实际 `total_cost` 为准 |

账号余额（memberships）：teamspace `seachenxyt` **balance ≈ $35** 量级（会变）。

## 3. S2 Mini 选定配置

| 项 | 选择 | 理由 |
|----|------|------|
| **机型** | **T4** | 标价 **~$0.19/h**，远低于 $3；S2 小 MoE 显存足够；同能力最便宜 |
| **备选** | L4（$0.48） | T4 排队/不可用时 |
| **高端备选** | A100（$2.19） | 仅当需要更大 batch/更快墙钟且仍 &lt;$3 |
| **不用** | H100（$4.50） | 超预算 |
| **cloud** | 默认（不强制贵厂） | 同卡选能 submit 且最终更便宜的路径 |
| **interruptible** | **false（不可中断）** | on-demand，避免 Pending 抢不到 / 中途被踢 |
| **max_runtime** | 3h | 费用上限约 `0.19×3 ≈ $0.57`（T4 on-demand 量级） |

**不选 A100/H100 做默认**：S2 模型仍小，T4 更符合「同需求最便宜」。

## 4. S2 Mini 训练规格

| 项 | 值 |
|----|-----|
| config | `002/configs/mini.yaml` |
| 模型 | 6 层 / 384 dim / 6 head / **8 expert top-2** |
| 数据 | 公开中文维基切片 **~250MB**（失败则 sample 兜底） |
| steps | **5000** |
| 有效 batch | 32 × grad_accum 2 |
| seq | 512 |
| vocab | 16k SP |
| 产物 | `002/artifacts/mini/` |

## 5. 命令

```bash
cd LightningAI-Lab

# 本地（可选短测）
python main.py 002 prepare --config mini
python main.py 002 train --config mini --max-steps 100

# 远程 S2（T4）
python main.py 002 job --config mini --machine T4 --max-steps 5000
```

环境变量：`LIGHTNING_USER_ID` / `LIGHTNING_API_KEY`，可选 `LIGHTNING_CLOUD=GCP|AWS`。  
**全程 on-demand（不可中断）；代码不再支持 spot/可中断提交。**
