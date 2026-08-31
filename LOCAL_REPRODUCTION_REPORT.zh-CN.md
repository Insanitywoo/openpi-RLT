# openpi-RLT 本地复现报告

> 报告日期：2026-08-31
> 
> 本报告记录当前工作区在本地 RTX 5070 Ti 上的实际复现结果。目标是先验证软件安装、导入、算法单元测试和在线 RL 子系统，而不是声称已经完成大模型训练或真实机器人复现。

## 1. 执行环境

| 项目 | 实际结果 |
|---|---|
| 操作系统 | Linux（当前工作站） |
| Python（根项目） | 3.11.16 |
| Python（在线 RL） | 3.10.21 |
| uv | 0.12.5 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| GPU 显存 | 16303 MiB |
| NVIDIA 驱动 | 580.173.02 |
| 根项目 JAX | 0.5.3 + CUDA 12 plugin |
| 根项目 PyTorch | 2.7.1+cu126 |
| 工作区状态 | 分支 `main`，领先 `origin/main` 1 个提交 |

项目实际采用的是 **uv**，不是 Conda：

```text
根目录 .venv              Python 3.11.16，OpenPI/VLA/RLT 主环境
rlt_online_rl/.venv       Python 3.10.21，在线 Actor/Critic/RL 环境
```

## 2. 已完成事项

### 2.1 锁文件检查

```bash
uv lock --check
```

结果：通过。只出现 `tool.uv.dev-dependencies` 弃用提示，不影响解析。

### 2.2 根项目依赖安装

```bash
uv sync --locked
```

结果：成功安装 241 个包，包括：

- JAX 0.5.3、Flax 0.10.2、Optax；
- PyTorch 2.7.1、CUDA 12 运行时组件；
- OpenPI、openpi-client、LeRobot；
- Orbax checkpoint、Transformers、W&B；
- pytest、ruff 等开发工具。

注意：首次安装耗时约 23 分钟，主要时间用于下载 PyTorch 和 CUDA 组件；系统盘/缓存大约增加数 GB。

### 2.3 根项目导入与设备发现

关键结果：

```text
imports: OK
torch CUDA available: True
torch GPU count: 1
torch GPU 0: NVIDIA GeForce RTX 5070 Ti
jax devices: [CudaDevice(id=0)]
```

这说明 Python 依赖和 JAX CUDA 后端可以工作，JAX 已经能够发现 GPU。

但 PyTorch 输出以下警告：

```text
RTX 5070 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

含义是当前 PyTorch wheel 声称只包含到 `sm_90` 的预编译 CUDA kernel，而 RTX 5070 Ti 的计算能力被识别为 `sm_120`。CUDA 设备仍然可以被发现，但某些 PyTorch CUDA 算子可能回退、报错或性能异常。这个问题必须在云端 Pro6000 上重新检查，不能直接把本机结果当成云端结果。

### 2.4 根项目轻量测试

执行：

```bash
uv run pytest -q \
  src/openpi/transforms_test.py \
  scripts/test_rlt_batch.py \
  scripts/test_rlt_client.py
```

结果：

```text
10 passed
```

这些测试验证了数据变换、RLT batch 处理和 RLT client 相关的轻量链路。

### 2.5 在线 RL 子环境与测试

创建并安装独立环境：

```bash
uv venv --python 3.10 rlt_online_rl/.venv
uv pip install --python rlt_online_rl/.venv/bin/python \
  -e packages/openpi-client \
  -e 'rlt_online_rl[dev]'
```

执行：

```bash
rlt_online_rl/.venv/bin/python -m pytest -q rlt_online_rl/tests
```

初次结果为 `41 passed, 2 failed`。失败原因不是模型数学错误，而是 `ActorService` 启动后台 snapshot 轮询线程存在竞态：测试在创建服务后很快发起请求，服务还没有来得及加载已有 actor snapshot，于是错误地返回 `ref_chunk` fallback。

已修复：在启动后台轮询线程之前，同步加载一次已有 snapshot：

```python
self._try_reload_snapshot()
self._start_param_poller()
```

修复后结果：

```text
43 passed
```

这项修复保证了：

1. 服务构造完成后，已有 snapshot 会立即可用；
2. 后台线程仍然负责后续 actor 参数热更新；
3. 首次请求不依赖线程调度时序。

## 3. 未完全通过的部分

### 3.1 完整根项目 pytest

执行完整测试时，结果中出现：

```text
5 failed, 10 passed
```

失败集中在：

```text
src/openpi/models/model_test.py
```

错误为 JAX GPU 显存不足：

```text
XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory
```

此外完整测试还触发了 `pi0_aloha_sim` checkpoint 下载；该下载是测试/模型初始化路径的一部分，并非普通 import 测试。测试运行约 12 分钟后，为避免继续消耗资源而手动中断，故完整 suite 不能记录为全通过。

这不是“环境没有安装成功”，而是 **16 GiB 级别本地 GPU 无法承载当前 Pi0 模型测试的初始化/编译峰值**。正式云端复现应在 Pro6000 上重新执行，并根据显存情况配置 JAX 内存比例。

### 3.2 `scripts/test_batch_padding.py`

其中两个测试需要 `client` fixture：

```text
fixture 'client' not found
```

该脚本不是当前 pytest 配置下可直接运行的完整测试文件，属于测试组织/fixture 缺失问题。不能把它解释为 RLT 算法失败。

### 3.3 Ruff

对修改文件执行 ruff 时发现该文件原有 11 个 lint 问题，主要是：

- 访问内部成员 `_packer`；
- 原有 `try/except/pass` 风格；
- 闭包捕获循环变量；
- 原有 zip strict、嵌套 if、列表推导等风格规则。

本次只修复了会导致测试不稳定的 snapshot 初始加载竞态，没有顺手重构这些与本次复现目标无关的历史 lint 问题。

## 4. 当前结论

### 已确认

- uv 根环境可以稳定创建；
- OpenPI/RLT 依赖可以安装；
- JAX CUDA 后端可以发现本机 GPU；
- 轻量 RLT 数据/客户端测试通过；
- 在线 RL 独立 Python 3.10 环境可以创建；
- 在线 RL 测试修复后 43 项全部通过；
- Machine B 的 actor snapshot 加载、推理、热更新链路基本可用。

### 尚未确认

- RTX 5070 Ti 上的完整 Pi0 模型初始化和训练；
- RLT 阶段 1 的真实 checkpoint 训练；
- 公开 LeRobot 数据训练；
- Machine A 的真实 VLA/RLT 服务；
- ManiSkill 环境适配；
- 多 GPU 分工和云端正式训练。

### 建议结论

本地阶段已经达到“环境安装成功 + 轻量软件链路跑通 + 在线 RL 单测通过”的验收标准。由于本机显存和 PyTorch 架构支持存在限制，正式 VLA/RLT 训练应转到百舸开发机；本地继续承担源码阅读、配置修改、单元测试和小型 fake/synthetic 验证。
