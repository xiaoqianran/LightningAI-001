# LightningAI-001

Lightning AI 平台学习仓库：从 **Batch Job 基础提交** 开始，对照官方文档逐步练习 Studio / Job / Deployment。

默认 teamspace：`seachenxyt/seachenxyt`（可用环境变量覆盖）。

## 目录

```text
LightningAI-001/
├── README.md
├── requirements.txt
├── docs/
│   └── 01-basic-submit.md      # 第一课说明
├── examples/
│   └── 01_hello_job.py         # SDK：提交 → 等待 → 日志
└── scripts/
    └── 01_hello_job.sh         # CLI：等价提交
```

## 快速开始

### 1. 安装

```bash
cd LightningAI-001
pip install -r requirements.txt
# 或已用 uv tool install lightning-sdk，确保 PATH 含 lightning
```

### 2. 登录

```bash
export LIGHTNING_USER_ID=...
export LIGHTNING_API_KEY=...
# 或
lightning login
```

### 3. 第一课：基础提交（Hello CPU Job）

**推荐（会等待并打印日志）：**

```bash
python examples/01_hello_job.py
```

**仅 CLI 提交：**

```bash
bash scripts/01_hello_job.sh
```

详见 [docs/01-basic-submit.md](docs/01-basic-submit.md)。

## 课程规划

| # | 主题 | 状态 |
|---|---|---|
| 01 | 基础提交（Docker + CPU hello） | ✅ |
| 02 | Studio Job | 待写 |
| 03 | GPU Job + 产物 | 待写 |
| 04 | 超参扫描 / 流水线 | 待写 |
| 05 | 多机 MMT | 待写 |

## 参考

- 本地文档索引：docs-mcp `library=lightning-ai`（Batch Jobs / CLI / SDK）
- 在线：[Batch jobs](https://lightning.ai/docs/platform/inference/batch-jobs)
- 官方 skills：`Lightning-AI/skills` → `lightning-jobs`
