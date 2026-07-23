# S1 Smoke 设计说明

> 阶段目标：在 Lightning 上用 **一次（可 resume）GPU Job** 证明  
> **中文数据 → 最小 MoE → 训练下降 → ckpt 可加载 → 能 dump 几句简体**  
> 全部跑通，费用与时间可控。  
> **不追求** 流畅对话；那是 S2。

关联：`docs/02-chinese-moe-train-plan.md`

---

## 1. 范围与成功标准

### 1.1 In scope

| 项 | 内容 |
|----|------|
| 模型 | 极小 MoE LM（配置 `smoke`，见 §3） |
| 数据 | 公开简体中文 **D0** 子集（§4） |
| 训练 | 单卡、固定 step、带 load-balance loss |
| 工程 | Docker Job 或等价；**ckpt + resume**；metrics JSON |
| 生成 | 训练结束对 3～5 条中文 prompt 做 greedy/sample 解码并写入 artifacts |

### 1.2 Out of scope（S1 不做）

- 大规模语料、多机、ZeRO、复杂并行  
- RLHF / SFT 指令数据  
- 与 dense 基线严格对比（可只记录 MoE 路由）  
- 调到「能聊」  

### 1.3 Definition of Done（全部满足才算 S1 通过）

| ID | 标准 | 如何判定 |
|----|------|----------|
| D1 | Job 终态 `Completed`（或 resume 拼接后等价完成） | Lightning status |
| D2 | `train/loss` 相对前 10% steps 的均值，**末 10% 下降 ≥ 15%**（或绝对下降明显） | `metrics.json` |
| D3 | 无持续 NaN/Inf | logs + metrics |
| D4 | 至少一个 `checkpoints/step_*.pt` 可 `torch.load` | `generate` 脚本 |
| D5 | `generations_smoke.txt` 含简体输出（允许胡言，但须是 UTF-8 中文汉字为主） | 人工扫一眼 |
| D6 | `router_stats.json` 无单 expert 占用 **> 95%** 全程（允许早期偏斜） | 统计 |
| D7 | 单价 **&lt; $3/h**；同卡选便宜商家 | 提交记录 |

---

## 2. 总体流程

```text
[本地或 Job 内]
  prepare_d0 → train_tokenizer → tokenize → train.bin / val.bin
       ↓
[Lightning Job S1]
  load config smoke
  (optional) resume from last ckpt
  train max_steps
  save ckpt + metrics + router stats
  run generate → generations_smoke.txt
       ↓
[本地验收]
  拉 logs / artifacts 或看 Job UI
  勾 DoD 清单
```

**原则：** S1 尽量 **一个 Job 内完成**「小数据准备 + 训练 + 生成」，减少来回；  
若准备太慢，可拆 `prepare` CPU Job，但默认一体。

---

## 3. 模型配置 `smoke`

比总计划 v0 **再小一档**，保证任意 &lt;$3/h GPU 都秒级 iteration。

| 超参 | 值 | 备注 |
|------|-----|------|
| `n_layer` | 4 | |
| `n_embd` | 256 | |
| `n_head` | 4 | head_dim=64 |
| `n_expert` | 4 | |
| `top_k` | 1 | 实现简单 |
| `expert_ffn_mult` | 2 | expert 隐层 = 2 * n_embd |
| 共享 | 仅 Attention + embed/lm_head dense | FFN 全 MoE |
| `vocab_size` | 8000 | SentencePiece unigram |
| `block_size` / seq | 256 | |
| 位置编码 | 学习绝对位置 or RoPE（二选一，实现选 **learned** 更简单） | |
| 归一化 | RMSNorm 或 LayerNorm | LayerNorm 更常见 |
| 激活 | GELU / SiLU | |
| 参数量级 | 约 **5M–15M**（含全部 expert 权重） | 激活更少 |

**损失：**

```text
L = L_ce + λ * L_balance
λ = 0.01   # smoke 固定
L_balance: Switch-Transformer 风格 aux loss（基于 router 概率与 token 分配）
```

**精度：** `bf16` if `cuda` 且 device 支持，else `fp16` + GradScaler，else `fp32`（CPU 单测）。

---

## 4. 数据设计 D0

### 4.1 来源（公开、可脚本下载）

**主选（实现简单、许可清晰）：**

| 优先级 | 来源 | 用法 |
|--------|------|------|
| 1 | **HuggingFace `wikimedia/wikipedia`** 中文 dump 切片 | `zh` / `20231101.zh` 等，只取前 N 篇纯文本 |
| 2 | 备选 **`PleIAs/Chinese-Commons`** 或类似开源中文段落 | 若 wiki 拉取失败 |
| 3 | 仓库内 **`data/sample/zh_smoke.txt`**（很小，&lt;1MB） | 离线/断网兜底，保证可测 |

S1 **不**依赖用户私有盘。

### 4.2 规模与过滤

| 项 | 值 |
|----|-----|
| 原始抓取上限 | **~30MB** 纯文本 或 **5 万行**（先到先停） |
| 行过滤 | 去空行；长度 10–2000 字；去掉明显 URL 行 |
| 简体 | `zhconv` 或 opencc **t2s**（可选依赖；若过重可仅过滤含大量繁体特征行） |
| 划分 | train **98%** / val **2%**，`seed=42` |
| 落盘 | `data/d0/train.txt`, `data/d0/val.txt` → tokenize → `train.bin`, `val.bin`（uint16/uint32 token id） |

### 4.3 Tokenizer

| 项 | 值 |
|----|-----|
| 算法 | SentencePiece **unigram** |
| vocab | 8000 |
| 语料 | 仅 D0 train.txt |
| 特殊符号 | `<pad>=0`, `<unk>`, `<s>`, `</s>`（按 SP 默认 + 文档化） |
| 产物 | `artifacts/tokenizer/spm_zh_8k.model` + `.vocab` |

S2 可 **复用** 同一 tokenizer，或重新在更大语料上训（S2 设计再定）；S1 只保证自洽。

---

## 5. 训练超参 `smoke`

| 项 | 值 |
|----|-----|
| `max_steps` | **2000** |
| `batch_size` | 32（OOM 则 16→8） |
| `grad_accum` | 1（OOM 时升到 2–4 保持 ~等效 batch） |
| `lr` | 3e-4 |
| `weight_decay` | 0.1 |
| `betas` | (0.9, 0.95) |
| `warmup_steps` | 50 |
| `lr_schedule` | cosine → 0.1 * lr |
| `grad_clip` | 1.0 |
| `log_every` | 20 steps |
| `eval_every` | 200 steps（val loss） |
| `ckpt_every` | **200 steps** 与 **每个 eval 后** |
| `save_last` | 始终保留 `last.pt` |
| `seed` | 42 |
| `num_workers` | 2 |

**过拟合自检（可选开关 `--overfit_one_batch`）：**  
仅用于本地 S0：同一 batch 重复 200 step，loss 应接近 0。S1 Job **默认关闭**。

### 5.1 时间盒

| 项 | 值 |
|----|-----|
| 预计纯训练 | 5–20 min |
| Job `max_runtime` | **3600 s（60 min）** |
| 超时 | `stop_on_timeout=True`；依赖 **last.pt resume** 可再提 Job |

### 5.2 GPU

```text
候选 = {A100, H100, RTXP_6000} ∩ {报价 < $3/h} ∩ {有货}
提交 = argmin 单价(候选)；空则 fallback 任意 < $3/h GPU
interruptible = False（默认 on-demand / 不可中断）
```

提交记录写入 `artifacts/run_meta.json`：machine、cloud、interruptible、单价（若可知）、job name、git commit。

---

## 6. Checkpoint 与 Resume

### 6.1 文件

```text
artifacts/checkpoints/
  last.pt                 # 始终覆盖
  step_000200.pt
  step_000400.pt
  ...
```

### 6.2 `last.pt` 内容

```python
{
  "model": state_dict,
  "optimizer": state_dict,
  "scaler": optional,
  "step": int,
  "config": dict,          # smoke 超参快照
  "tokenizer_path": str,
  "metrics": {...},        # 最近 val 等
  "rng": optional,
}
```

### 6.3 Resume 行为

```bash
python -m moe_zh.train --config configs/smoke.yaml --resume auto
# auto: 若 ARTIFACT_DIR/checkpoints/last.pt 存在则加载，step 续跑到 max_steps
```

- Job 被抢占 → 同命令再提交 → 从 `last.pt` 继续直到 `max_steps`  
- **不**自动提高 `max_steps`；续跑只补齐剩余 steps  

### 6.4 Artifacts 根目录

Job 内环境变量：

```text
MOE_ARTIFACT_DIR=/teamspace/jobs/<job_name>/artifacts
# Docker 无 teamspace 时：
MOE_ARTIFACT_DIR=/workspace/artifacts   # 再 lightning cp 或挂 path_mapping
```

**S1 推荐：** Studio Job 用 teamspace 路径；或 Docker + 结束时把 artifacts 打成 tar 打日志 URL。  
实现首选：**可写本地 `./artifacts`，Job 命令结束前 `cp -r` 到 teamspace（若存在）**。

---

## 7. 代码与仓库布局（S1 要实现的）

```text
LightningAI-001/
  configs/
    smoke.yaml                 # 唯一 S1 配置源
  moe_zh/
    __init__.py
    config.py                  # load yaml
    model.py                   # MoE LM
    data.py                    # bin dataset + prepare hooks
    tokenizer_util.py
    prepare_d0.py              # 下载/过滤/SP/bin
    train.py                   # 主循环 + resume
    generate.py
    balance.py                 # aux loss
  scripts/
    s1_smoke_local.sh          # CPU/GPU 本地冒烟（可选）
    s1_smoke_job.py            # 提交 Lightning Job（SDK）
  data/
    sample/zh_smoke.txt        # 极小兜底
    d0/                        # gitignore 大文件；可留 README
  docs/
    03-s1-smoke-design.md      # 本文
  requirements-train.txt       # torch, sentencepiece, datasets, pyyaml, ...
```

### 7.1 模块职责

| 模块 | 职责 |
|------|------|
| `model.py` | `MoELanguageModel`：embed、N×(Attn+MoE-FFN)、lm_head；返回 logits + router 统计 |
| `balance.py` | `load_balancing_loss(router_probs, expert_indices)` |
| `prepare_d0.py` | CLI：产出 `data/d0/*` + tokenizer |
| `train.py` | DDP 不做；单卡循环；写 metrics / ckpt |
| `generate.py` | 加载 ckpt+tokenizer；prompts 列表；写 `generations_smoke.txt` |
| `s1_smoke_job.py` | 选 machine（预算内最便宜逻辑可先手动常量）、`Job.run`、wait、打印 link |

### 7.2 Job 命令（逻辑）

```bash
set -e
pip install -r requirements-train.txt   # 或镜像预装
python -m moe_zh.prepare_d0 --out data/d0 --max-bytes 30000000
python -m moe_zh.train --config configs/smoke.yaml --resume auto
python -m moe_zh.generate --ckpt artifacts/checkpoints/last.pt --out artifacts/generations_smoke.txt
```

镜像建议：`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`（或当时稳定 tag）  
**不要**用 `python:3.11-slim` 现编 torch。

---

## 8. 监控与产物清单

训练结束 `artifacts/` 应有：

```text
artifacts/
  run_meta.json
  metrics.json           # steps[], train_loss[], val_loss[], lr[]
  router_stats.json      # per-expert token counts / fractions
  tokenizer/
  checkpoints/last.pt
  checkpoints/step_*.pt  # 至少 1 个阶段性
  generations_smoke.txt
  train.log              # stdout tee 可选
```

### 8.1 生成 prompt 固定集（简体）

```text
今天天气
人工智能是
北京是
请写一句问候
春天来了
```

解码：`temperature=0.8`, `top_k=50`, `max_new_tokens=64`；若全是 unk 则 fail D5。

---

## 9. 验收步骤（人工 5 分钟）

1. 打开 Job link，确认 machine 与 **Completed**  
2. 下载或查看 `metrics.json`：loss 曲线下降  
3. 查看 `router_stats.json`：四 expert 均有计数  
4. 打开 `generations_smoke.txt`：有汉字  
5. 记录：墙钟、`$`、是否 resume 过  

**S1 通过 → 再开 S2 设计/实现。**  
失败则按 §10 排障，不扩大数据。

---

## 10. 风险与排障

| 现象 | 处理 |
|------|------|
| 拉 wiki 失败 | 回退 `data/sample` + 重复采样扩增到够 step |
| OOM | batch 减半；seq 保持 256 |
| loss 不降 | 开本地 overfit_one_batch；查 lr/标签 shift |
| 单 expert 吃满 | 提高 λ 到 0.02–0.05；检查 router 初始化 |
| Pending &gt; 20 min | 换同预算更便宜/有货机型；勿升价超 $3 |
| 抢占 | 同命令 resume；确认 last.pt 已上传/持久路径 |
| 生成乱码 | tokenizer 与 ckpt 路径不一致；encoding utf-8 |

---

## 11. 实现顺序（S1 工程任务拆分）

| 序 | 任务 | 估计 | 依赖 |
|----|------|------|------|
| 1 | `configs/smoke.yaml` + `config.py` | 0.5h | — |
| 2 | `model.py` + `balance.py` + CPU 一步 forward/backward | 2–4h | 1 |
| 3 | `prepare_d0.py` + sample 兜底 | 1–2h | — |
| 4 | `train.py` + ckpt/resume + metrics | 2–3h | 2–3 |
| 5 | `generate.py` | 0.5–1h | 4 |
| 6 | `requirements-train.txt` + 本地 CPU 短跑 | 1h | 4–5 |
| 7 | `s1_smoke_job.py` + 真 GPU Job | 1h | 6 |
| 8 | 验收 DoD、改 README | 0.5h | 7 |

**合计实现量级：约 1–1.5 人日**（含一次 Job 翻车重跑）。

---

## 12. 与 S2 的接口（预留）

S1 成功后冻结：

- `configs/smoke.yaml` 作为最小可运行参考  
- tokenizer 可被 S2 `configs/mini.yaml` 引用或重训  
- `train.py --resume` 契约不变  
- S2 只改：数据规模、`max_steps`、模型 width/depth、时间盒  

---

## 13. 提交前确认清单（每次 S1 Job）

```text
[ ] machine ∈ 允许集或 fallback，报价 < $3/h
[ ] 同卡已选更便宜商家（若可选）
[ ] interruptible 与 resume 路径一致（持久 artifacts）
[ ] max_runtime = 3600
[ ] max_steps = 2000，config = smoke
[ ] 数据 = 公开 D0 / sample 兜底
[ ] 命令含 train + generate
```

---

## 14. 一句话总结

**S1 = 4 层 256 维、4 expert top-1 的中文小 MoE，在 ≤30MB 公开中文上训 2000 step，单卡 &lt;$3/h（可抢占+resume），60 分钟内交出：下降的 loss、可加载 ckpt、几句简体生成、路由统计。**

---

**下一步（实现）：** 按 §11 从 `configs/smoke.yaml` + `moe_zh/model.py` 开始写代码；你点头后开工。
