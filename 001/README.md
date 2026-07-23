# 001 — Hello Lightning Job

最小远程冒烟：提交一个 CPU Docker Job，打印 Hello 后结束。

## 用法

```bash
# 仓库根目录
python main.py 001 run

# 或
cd 001 && python run.py run
```

## 依赖

```bash
pip install lightning-sdk
export LIGHTNING_USER_ID=...
export LIGHTNING_API_KEY=...
```

## 说明

- 镜像：`python:3.11-slim`
- 机器：CPU
- 默认 teamspace：`seachenxyt/seachenxyt`
