# 003 — NVIDIA NIM Agent（最小可跑）

用 **NVIDIA NIM**（OpenAI 兼容 API）当大脑，带 **白名单工具** 执行任务。

## 安全

```bash
export NVIDIA_API_KEY='nvapi-...'   # 勿提交到 git
# 可选
export NIM_MODEL='meta/llama-3.1-8b-instruct'
export NIM_BASE_URL='https://integrate.api.nvidia.com/v1'
```

## 用法

```bash
# 仓库根目录
python main.py 003 run --task "列出 002 目录并说明 S2 怎么提交"

# 或
cd 003
python run.py run --task "现在几点？读一下 README 前 20 行"
python run.py chat                 # 多轮（简单）
python run.py ping                 # 测 API 是否通
```

## 工具（白名单）

| 工具 | 作用 |
|------|------|
| `list_dir` | 列仓库内相对路径 |
| `read_file` | 读文本文件（限大小） |
| `run_shell` | 仅允许前缀白名单命令 |
| `lab_help` | 返回 LightningAI-Lab 用法摘要 |
| `sandbox_run` | **在远程 Lightning Sandbox 执行 bash**（需 `LIGHTNING_SANDBOX_ID`） |

**不会** 默认提交 Lightning Job。

## 远程 Sandbox

```bash
# 需要 org-scoped key（org 名是 seachenxyt-org，不是 seachenxyt）
export LIGHTNING_SANDBOX_API_KEY=$(lightning api-key get --org seachenxyt-org)

export LIGHTNING_SANDBOX_ID=$(python main.py 003 sandbox-create --name lab-agent | head -1)
python main.py 003 sandbox-probe --id "$LIGHTNING_SANDBOX_ID"
python main.py 003 run --task "在 sandbox 里执行 python --version"
python main.py 003 sandbox-delete --id "$LIGHTNING_SANDBOX_ID"
```

能力上限见 `docs/003-sandbox-ceiling.md`。

## 依赖

```bash
pip install requests
# 可选: openai 客户端风格也可，本实验用 requests
```
