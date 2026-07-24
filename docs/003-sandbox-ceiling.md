# 003 Sandbox + NIM Agent 能力实测

日期：2026-07-24  
账号 org：`seachenxyt-org` · teamspace：`seachenxyt/seachenxyt`

## 1. 鉴权

| 项 | 结果 |
|----|------|
| 个人 `LIGHTNING_API_KEY` | ❌ 不能创建 Sandbox（要求 org/teamspace-scoped key） |
| `lightning api-key create --org seachenxyt-org` | ✅ 得到 `sk-lit-...` → `LIGHTNING_SANDBOX_API_KEY` |
| 组织显示名 | membership 里 teamspace 名 `seachenxyt`，**org name 是 `seachenxyt-org`** |

## 2. 成功创建

```text
id: 01ky9b4x4svp5fc3yytz512p6t
name: lab-agent-probe
instance_type: cpu-1
runtime: python313
status: running
timeout: 900000 ms (15 min)
persistent: false
spot: false
```

## 3. 实测上限（cpu-1 + python313）

| 维度 | 实测 |
|------|------|
| OS | Ubuntu 24.04 · **gVisor** 内核 `Linux sandbox 4.19.0-gvisor` |
| CPU | **2 vCPU** |
| 内存 | **~1.2 GiB** |
| 根盘 | **~5.0 G**（几乎全空） |
| 用户 | **root** |
| Python | **3.13.1** |
| 网络出口 | ✅ `https://example.com` → 200 |
| pip | ✅ 可装包（如 requests） |
| GPU | ❌ 无 `nvidia-smi`（cpu-1 无 GPU） |
| 启动速度 | 创建到 running **约数秒** |
| 文件系统 | 可写；工作目录需自建（如 `/tmp/ws`） |

## 4. 和 Job / Studio 的边界

| | Sandbox | Job (我们 002) | Studio |
|--|---------|----------------|--------|
| 起速 | 秒级 | 分钟级 | 分钟级 |
| 算力 | 小 CPU（本测 2c/1.2G） | T4 等 GPU | 可选 GPU |
| 用途 | Agent 执行代码、试错 | 训练批处理 | 长期开发 |
| 持久 | 默认临时；可 persistent+快照 | Job 结束容器销毁 | 磁盘可持久 |
| 鉴权 | 需 org/teamspace sandbox key | 个人 API key 即可 | 个人登录即可 |

## 5. Agent 技能栈（003）

| 技能/工具 | 作用 | 上限备注 |
|-----------|------|----------|
| NIM 对话 | 规划、解释 | 受模型与速率限制 |
| list_dir / read_file | 读 Lab 仓库 | 仅仓库内路径 |
| run_shell | 本机白名单 shell | 不能任意命令 |
| **sandbox_run** | **远程隔离执行** | 依赖 `LIGHTNING_SANDBOX_ID` + sandbox key |
| lab_help | 实验用法摘要 | 静态文本 |

**Agent 上限 ≈ NIM 推理质量 × 工具覆盖面 × Sandbox 机器规格。**  
当前未接：GPU sandbox、自动交 Job、任意网络外的特权、持久大磁盘。

## 6. 推荐用法

```bash
# 1) org key（只放环境变量）
export LIGHTNING_SANDBOX_API_KEY=$(lightning api-key get --org seachenxyt-org)

# 2) 创建 sandbox
export LIGHTNING_SANDBOX_ID=$(python main.py 003 sandbox-create --name lab-agent)

# 3) NIM agent 在 sandbox 里跑命令
export NVIDIA_API_KEY=nvapi-...
python main.py 003 run --task "在 sandbox 里 pip 安装 httpx 并打印版本"

# 4) 用完删除（避免计费）
python main.py 003 sandbox-delete --id "$LIGHTNING_SANDBOX_ID"
```

## 7. 结论：能做什么 / 不能做什么

**能：** 给 NIM Agent 一个远程、隔离、可 pip、可联网的 Python 3.13 小 VM，做试跑、判题、轻量脚本、agent 工具执行。  

**不能（当前规格）：** 训 GPU 模型、当 Studio 用、当长期服务器、用个人 login key 直接创建。
