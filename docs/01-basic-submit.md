# 01 — 基础提交（Hello Job）

第一个练习：在 Lightning AI 上提交一个 **CPU Docker Job**，打印一行输出后结束。

不依赖任何 Studio 内容，只验证：鉴权 → 提交 → 等待 → 读日志。

## 前置条件

1. 已安装依赖：

   ```bash
   pip install -r requirements.txt
   # 或：uv tool install lightning-sdk
   ```

2. 已登录（二选一）：

   ```bash
   lightning login
   # 或
   export LIGHTNING_USER_ID=...
   export LIGHTNING_API_KEY=...
   ```

3. 默认 teamspace（本仓库示例默认值）：

   | 项 | 值 |
   |---|---|
   | owner | `seachenxyt` |
   | teamspace | `seachenxyt` |

   可用环境变量覆盖：`LIGHTNING_ORG` / `LIGHTNING_USER`、`LIGHTNING_TEAMSPACE`。

## 方式 A：Python SDK

```bash
cd LightningAI-001
python examples/01_hello_job.py
```

脚本会：

1. `Job.run(...)` 提交
2. `job.wait(...)` 等到终态
3. 打印 `status`、费用、logs

## 方式 B：CLI

```bash
cd LightningAI-001
bash scripts/01_hello_job.sh
```

等价命令：

```bash
lightning job run \
  --name hello-cpu-<timestamp> \
  --teamspace seachenxyt \
  --org seachenxyt \
  --image python:3.11-slim \
  --machine CPU \
  --command 'python -c "print(\"Hello from Lightning Job!\")"'
```

列出 / 停止 / 删除：

```bash
lightning job list --teamspace seachenxyt/seachenxyt
lightning job inspect <name> --teamspace seachenxyt/seachenxyt
lightning job stop <name> --teamspace seachenxyt/seachenxyt
lightning job delete <name> --teamspace seachenxyt/seachenxyt
```

## 官方概念对应

| 概念 | 本示例取值 |
|---|---|
| 环境 | Docker image（`python:3.11-slim`） |
| 机器 | `CPU`（最便宜，适合 smoke test） |
| 命令 | 一行 `print` |
| 异步 | SDK `Job.run` 立即返回，再 `wait` |

下一课可做：Studio Job、GPU Job、产物路径、超参扫描。
