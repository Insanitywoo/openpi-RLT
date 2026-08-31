# openpi-RLT 本地复现教程

本教程对应 `LOCAL_REPRODUCTION_REPORT.zh-CN.md` 的实际执行过程。今后迁移到百舸开发机时，可以把路径替换成云端工作目录，并重新执行同样的检查。

## 1. 为什么采用两个 uv 环境

仓库包含两个 Python 版本边界不同的部分：

```text
openpi 主项目       requires-python >= 3.11
rlt_online_rl       requires-python >= 3.10,<3.11
```

因此不把两者强行装进同一个环境：

```bash
# 根项目：Python 3.11
cd ~/projects/openpi-RLT
source .venv/bin/activate

# 在线 RL：Python 3.10
source rlt_online_rl/.venv/bin/activate
```

也可以完全不激活环境，显式调用 `uv run` 或环境内 Python。显式调用更不容易混淆：

```bash
uv run python --version
rlt_online_rl/.venv/bin/python --version
```

## 2. 根项目环境初始化

### 2.1 只检查锁文件

```bash
cd ~/projects/openpi-RLT
uv lock --check
```

`--check` 只检查 `pyproject.toml` 和 `uv.lock` 是否一致，不安装依赖，也不改锁文件。

### 2.2 按锁文件安装

```bash
uv sync --locked
```

`--locked` 的作用是禁止 uv 重新解析依赖版本。对于复现项目，这比无条件 `uv sync` 更适合，因为可以避免今天安装的版本和以后安装的版本漂移。

### 2.3 导入检查

```bash
uv run python - <<'PY'
import jax
import flax
import torch
import numpy
import optax
import yaml
import openpi
import openpi_client
import orbax
import transformers
import wandb

print("jax:", jax.__version__)
print("flax:", flax.__version__)
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("torch CUDA available:", torch.cuda.is_available())
print("torch GPU count:", torch.cuda.device_count())
print("jax devices:", jax.devices())
PY
```

成功标准：没有 ImportError，并且 JAX 能列出 CUDA device。

## 3. 在线 RL 独立环境

```bash
cd ~/projects/openpi-RLT
uv venv --python 3.10 rlt_online_rl/.venv
uv pip install --python rlt_online_rl/.venv/bin/python \
  -e packages/openpi-client \
  -e 'rlt_online_rl[dev]'
```

运行测试：

```bash
rlt_online_rl/.venv/bin/python -m pytest -q rlt_online_rl/tests
```

成功标准：当前应为：

```text
43 passed
```

## 4. 推荐的测试顺序

不要一开始就直接运行完整 `pytest`。完整 suite 可能下载模型并初始化大 Pi0 网络，16 GiB 显存机器容易 OOM。

推荐顺序：

```bash
# 第一层：纯数据和接口测试
uv run pytest -q \
  src/openpi/transforms_test.py \
  scripts/test_rlt_batch.py \
  scripts/test_rlt_client.py

# 第二层：在线 RL 全部单测
rlt_online_rl/.venv/bin/python -m pytest -q rlt_online_rl/tests

# 第三层：查看完整测试而不执行
uv run pytest --collect-only -q

# 第四层：正式云端再运行完整 suite
uv run pytest -q
```

## 5. 如何理解当前测试结果

### 5.1 JAX OOM 不等于环境安装失败

如果出现：

```text
XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory
```

先区分两种情况：

- import 阶段失败：通常是依赖、动态库或 CUDA 安装问题；
- 模型初始化/编译阶段失败：通常是显存、XLA 编译峰值或模型规模问题。

本机属于第二种。JAX 已经能看到 GPU，但 Pi0 模型测试的峰值超过了本机可用显存。

排查命令：

```bash
nvidia-smi
```

可在小型测试中临时使用：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
uv run pytest -q <某个小测试文件>
```

这不能保证完整 Pi0 模型一定能在本机运行，只是避免 JAX 一启动就预分配过多显存。

### 5.2 PyTorch 架构警告

如果看到 `sm_120 is not compatible`，说明当前 PyTorch wheel 没有为本机 GPU 架构提供对应预编译 kernel。不要在没有验证前强行改 CUDA/PyTorch 版本。正式云端需要重新记录：

```bash
nvidia-smi
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

### 5.3 测试中出现模型下载

某些 OpenPI 测试会通过 Google Cloud Storage 下载公开 checkpoint。首次运行较慢，并且会写入：

```text
~/.cache/openpi/
```

云端应把缓存位置放到持久化数据盘，避免开发机重建后重复下载。

## 6. 当前代码修复说明

文件：

```text
rlt_online_rl/src/rlt_online_rl/inference.py
```

修复前：服务构造时只启动后台轮询线程。测试或真实客户端若在轮询线程第一次运行前发请求，就会看到：

```text
actor_param_version = -1
refined_chunk = ref_chunk
```

修复后：构造服务时先同步尝试加载已有 snapshot，再启动后台线程。这样首次请求具备确定性，后续仍支持热更新。

## 7. 下一阶段执行顺序

本地已经完成基础环境复现后，后续顺序应是：

1. 提交并推送当前 `inference.py` 修复；
2. 在百舸开发机 clone 最新提交；
3. 执行云端 GPU、驱动、CUDA 和磁盘勘测；
4. 用 uv 重建两个环境；
5. 先跑 43 项在线 RL 测试和根项目轻量测试；
6. 云端运行完整 OpenPI 测试；
7. 再进行 RLT synthetic smoke test；
8. 最后才下载完整模型、数据集和启动训练。

代码提交命令由使用者执行：

```bash
git status
git diff --check
git add rlt_online_rl/src/rlt_online_rl/inference.py \
  LOCAL_REPRODUCTION_REPORT.zh-CN.md \
  LOCAL_REPRODUCTION_TUTORIAL.zh-CN.md
git commit -m "fix: load actor snapshot before serving"
git push origin main
```

推送之前应检查 diff，确认没有把本地缓存、checkpoint 或 `.venv` 加入 Git。
