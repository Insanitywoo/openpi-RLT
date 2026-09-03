# openpi-RLT 云端部署与完整训练指南

> **最后更新：** 2026-09-03  
> **适用目标：** 在具备充足 GPU 算力的百舸云开发机上，复现 openpi-RLT 的完整软件、RLT 阶段 1 训练和后续 Machine A / Machine B 链路。  
> **执行原则：** 先做可验证的单卡 smoke test，再扩大到多卡训练；不要跳过环境、数据与 checkpoint 验收步骤。

相关文档：

- [本地复现报告](LOCAL_REPRODUCTION_REPORT.zh-CN.md)
- [本地复现教程](LOCAL_REPRODUCTION_TUTORIAL.zh-CN.md)
- [RLT 理论参考](RLT_THEORY_REFERENCE.zh-CN.md)
- [长期复现计划](REPRODUCTION_PLAN.zh-CN.md)

---

## 1. 当前复现边界

本地已验证：根项目 uv 环境、JAX GPU 识别、10 项轻量 RLT 测试、`rlt_online_rl` 的 43 项单测，以及 ActorService 初始 snapshot 加载和热更新逻辑。

本地尚未完成：完整 Pi0/Pi0.5 模型测试、真实 RLT 数据训练、真实 Machine A 服务、Machine A/B 联调、ManiSkill 在线闭环。本机约 16 GiB 显存不足以稳定完成完整 Pi0 模型初始化测试，因此正式训练应转到云端。

---

## 2. 推荐云端资源与拓扑

推荐首选资源：4 GPU、32 CPU 核、128 GiB 内存、32 GiB 共享内存。具体 GPU 型号和每卡显存必须以创建后的 `nvidia-smi` 为准。

初期采用**一台云端开发机、多进程、多 GPU**；Machine A / Machine B 是逻辑角色，不需要两台物理服务器。

```text
GPU 0：Machine A，冻结 Pi0.5 + RLT 推理
GPU 1：Machine B ActorService 或 LearnerService
GPU 2：Machine B LearnerService 或 ManiSkill rollout
GPU 3：评估、数据预处理、额外 rollout 或备用
```

第一阶段不应立即占满四卡：先单卡跑环境检查、测试和 `debug_rlt`；确认后再试两卡 FSDP，最后再做四卡正式训练。

---

## 3. 存储与缓存规划

系统盘适合代码、uv 环境和少量日志。模型、数据、checkpoint、replay 与缓存必须写入持久化挂载盘。

以下假设持久化盘挂载到 `/mnt/openpi-rlt`；请按实际平台路径替换。

```text
/mnt/openpi-rlt/
├── datasets/       LeRobot / 自有数据
├── pretrained/     预训练权重或额外模型
├── checkpoints/    RLT 阶段 1 checkpoint
├── cache/
│   ├── huggingface/
│   └── openpi/
├── replay/         replay journal
├── runs/           learner checkpoint 与 actor snapshot
├── wandb/          W&B 日志
└── logs/           stdout/stderr 与诊断日志
```

创建目录及设置通用缓存变量：

```bash
export OPENPI_ROOT=/mnt/openpi-rlt
mkdir -p \
  "$OPENPI_ROOT/datasets" \
  "$OPENPI_ROOT/pretrained" \
  "$OPENPI_ROOT/checkpoints" \
  "$OPENPI_ROOT/cache/huggingface" \
  "$OPENPI_ROOT/cache/openpi" \
  "$OPENPI_ROOT/replay" \
  "$OPENPI_ROOT/runs" \
  "$OPENPI_ROOT/wandb" \
  "$OPENPI_ROOT/logs"

export HF_HOME="$OPENPI_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$OPENPI_ROOT/cache"
export WANDB_DIR="$OPENPI_ROOT/wandb"
```

建议保存为 `~/openpi-rlt-env.sh`：

```bash
cat > ~/openpi-rlt-env.sh <<'EOF_ENV'
export OPENPI_ROOT=/mnt/openpi-rlt
export HF_HOME="$OPENPI_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$OPENPI_ROOT/cache"
export WANDB_DIR="$OPENPI_ROOT/wandb"
EOF_ENV
source ~/openpi-rlt-env.sh
```

首次 OpenPI checkpoint 下载后，应检查实际缓存是否位于持久化盘；不要在未确认前盲目删除 `$HOME/.cache/openpi`。

---

## 4. 为什么先用 uv 而不是 Docker

首轮云端复现推荐直接使用 `uv`，不先构建 Docker：

1. 仓库已经有 `uv.lock`，可锁定 Python 依赖版本；
2. GPU 驱动来自云端宿主机，容器不能替代对实际驱动/JAX/CUDA 兼容性的验证；
3. 直接使用 uv 更便于观察每一步、定位 JAX、CUDA、数据或权重问题；
4. 在训练和服务链路稳定后，再用 Docker 固化已验证环境更合适。

镜像自带 Python 3.12 不构成阻碍：根项目由 uv 创建 Python 3.11 环境；`rlt_online_rl` 由 uv 创建独立 Python 3.10 环境。

---

## 5. 阶段 0：云端只读勘测

SSH 登录后先执行：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nproc
free -h
df -h
uname -a
python3 --version
uv --version
```

验收：记录 GPU 数、每卡显存、驱动、CPU、内存、持久化盘挂载点、Python 与 uv 版本。若 `uv` 不存在，安装 uv 前先确认网络与平台限制。

---

## 6. 阶段 1：同步代码和建立两个环境

### 6.1 获取代码

本地代码已推送后，在云端执行：

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone git@github.com:Insanitywoo/openpi-RLT.git
cd openpi-RLT
git log -2 --oneline
```

已有 clone 时：

```bash
cd ~/workspace/openpi-RLT
git pull --ff-only origin main
```

### 6.2 根 OpenPI/RLT 环境（Python 3.11）

```bash
cd ~/workspace/openpi-RLT
uv python install 3.11
uv venv --python 3.11 .venv
uv sync --locked
```

验证导入和 GPU：

```bash
uv run python - <<'PY'
import sys
import jax
import torch
import openpi
import openpi_client

print("Python:", sys.version)
print("JAX:", jax.__version__)
print("JAX devices:", jax.devices())
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.cuda.is_available())
print("Torch GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

成功标准：JAX 和 PyTorch 都能看到分配给开发机的全部 GPU。

### 6.3 在线 RL 环境（Python 3.10）

必须与根 `.venv` 隔离：

```bash
cd ~/workspace/openpi-RLT
uv python install 3.10
uv venv --python 3.10 rlt_online_rl/.venv
uv pip install \
  --python rlt_online_rl/.venv/bin/python \
  -e packages/openpi-client \
  -e 'rlt_online_rl[dev]'
```

---

## 7. 阶段 2：测试验收顺序

### 7.1 在线 RL 单测

```bash
cd ~/workspace/openpi-RLT
rlt_online_rl/.venv/bin/python -m pytest -q rlt_online_rl/tests
```

当前代码的成功标准：

```text
43 passed
```

### 7.2 根项目轻量测试

```bash
cd ~/workspace/openpi-RLT
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest -q \
  src/openpi/transforms_test.py \
  scripts/test_rlt_batch.py \
  scripts/test_rlt_client.py
```

成功标准：

```text
10 passed
```

### 7.3 完整根项目测试

仅在前两项通过后执行：

```bash
cd ~/workspace/openpi-RLT
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run pytest -q
```

完整测试可能下载公开 OpenPI checkpoint 并写入缓存。首次运行较慢是正常现象。若失败，先保存完整 traceback；不要在没有判断根因前反复重装依赖。

---

## 8. 阶段 3：RLT 阶段 1 smoke training

先跑仓库内置 fake-data 配置。它验证的不是策略效果，而是：Pi0.5/RLT 模型构建、VLA prefix、RLT 编码/解码、loss、反向传播和 checkpoint。

### 8.1 冻结 VLA 的 RLT smoke test

```bash
cd ~/workspace/openpi-RLT
source ~/openpi-rlt-env.sh

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train_rlt.py debug_rlt \
  --checkpoint-base-dir /mnt/openpi-rlt/checkpoints \
  --exp-name cloud-debug-rlt \
  --overwrite
```

`debug_rlt` 具有：dummy 模型、fake data、batch size 2、10 step、`rlt_alpha=0`，即冻结 VLA，只训练 RLT module。

验收：完成 10 step，打印 loss，且在下方位置写出 checkpoint：

```text
/mnt/openpi-rlt/checkpoints/debug_rlt/cloud-debug-rlt/
```

### 8.2 联合训练 smoke test

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train_rlt.py debug_rlt_joint \
  --checkpoint-base-dir /mnt/openpi-rlt/checkpoints \
  --exp-name cloud-debug-rlt-joint \
  --overwrite
```

区别：`debug_rlt_joint` 设置 `rlt_alpha=1.0`；总损失包含 RLT 重建项和 VLA 训练项，VLA 不再完全冻结。

---

## 9. 阶段 4：真实 RLT 数据训练

项目真实 AgileX 配置包括：

```text
rlt_pi05_agilexbag_image_delta
rlt_pi05_agilexbag_image_delta_joint
```

关键参数：

```text
基础模型：Pi0.5
动作维度：32
动作 horizon：50
数据动作形式：delta joint actions
RLT：1 个 token、2 层、2048 embedding/input dim
训练：默认 5000 steps、默认全局 batch size 32、8 data workers
```

### 9.1 数据前置条件

真实配置读取的 LeRobot 仓库由环境变量控制，默认值只是占位符，必须替换：

```bash
export AGILEX_LEROBOT_REPO="<你的 Hugging Face LeRobot 数据集 ID>"
export AGILEX_PI05_BASE_CKPT="gs://openpi-assets/checkpoints/pi05_base/params"
```

如果数据集私有，还需要按 Hugging Face 的认证方式配置 token。必须先确认数据包含该配置所需的相机、state、32 维动作和动作归一化统计；不匹配时先修数据 adapter / config，不能直接训练。

### 9.2 单卡真实 smoke training

先验证真实数据、基础权重和 checkpoint 结构：

```bash
cd ~/workspace/openpi-RLT
source ~/openpi-rlt-env.sh

export AGILEX_LEROBOT_REPO="<真实数据集 ID>"
export AGILEX_PI05_BASE_CKPT="gs://openpi-assets/checkpoints/pi05_base/params"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/train_rlt.py rlt_pi05_agilexbag_image_delta \
  --checkpoint-base-dir /mnt/openpi-rlt/checkpoints \
  --exp-name rlt-delta-smoke \
  --num-train-steps 20 \
  --batch-size 1 \
  --num-workers 2 \
  --wandb-enabled false
```

验收：基础 Pi0.5 权重加载、数据加载、loss 计算、20 step 训练和 checkpoint 保存均成功。

### 9.3 正式单卡和多卡训练

单卡 smoke 后，逐级增加 batch 和训练步数，例如 100、1000、5000 step。不要跳过每一级日志、显存和 checkpoint 检查。

配置支持 `fsdp_devices`。两卡 smoke 示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train_rlt.py rlt_pi05_agilexbag_image_delta \
  --checkpoint-base-dir /mnt/openpi-rlt/checkpoints \
  --exp-name rlt-delta-fsdp2-smoke \
  --batch-size 4 \
  --num-train-steps 20 \
  --fsdp-devices 2 \
  --wandb-enabled false
```

只有在 1 卡和 2 卡 smoke 均通过后，才扩大为 4 卡和正式 5000 step；全局 batch size 要根据实测显存和吞吐逐步确定。

---

## 10. 阶段 5：Machine A 部署

训练 checkpoint 完成后，Machine A 加载冻结 Pi0.5 + RLT，输出 `z_rl` 与 VLA `ref_chunk`。单机初期放在 GPU 0：

```bash
cd ~/workspace/openpi-RLT
source ~/openpi-rlt-env.sh

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
uv run python scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta \
  --checkpoint-dir /mnt/openpi-rlt/checkpoints/rlt_pi05_agilexbag_image_delta/<exp-name>/<step> \
  --port 8000 \
  --shared-prefix-inference
```

`--shared-prefix-inference` 仅复用 VLA prefix/KV cache 以降低推理延迟，不改变模型权重、训练或 online-RL payload。为严格保持旧推理路径，可省略此开关。

Machine A 和 B 在同一台云机时，使用：

```text
ws://127.0.0.1:8000
```

不建议把 8000、9101、9102 暴露到公网；需要远程调试时应使用 SSH tunnel。

---

## 11. 阶段 6：Machine B 部署

Machine B 由 `ActorService`、`LearnerService`、`ReplayManager` 构成，使用 Python 3.10 环境。真实 AgileX 配置位于：

```text
rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml
```

启动前应复制一份配置到云端实验目录，修改至少以下字段：

```yaml
runtime:
  env_driver:
    machine_a_ws_url: ws://127.0.0.1:8000

  learner_service:
    checkpoint_dir: /mnt/openpi-rlt/runs/<run-name>/checkpoints
    actor_snapshot_path: /mnt/openpi-rlt/runs/<run-name>/actor_snapshot/actor_snapshot.pkl

  replay:
    journal_path: /mnt/openpi-rlt/replay/<run-name>/replay_journal.pkl
```

启动：

```bash
cd ~/workspace/openpi-RLT/rlt_online_rl
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 \
python launch/launch_machine_b.py \
  --config <云端复制后的 online_rl.yaml>
```

真实 AgileX 配置依赖 ROS/机器人接口，因此在 ManiSkill adapter 尚未实现前，不应直接启动真实 rollout。Machine B 可先配合 fake Machine A 或单元测试验证：

```bash
cd ~/workspace/openpi-RLT/rlt_online_rl
source .venv/bin/activate
python launch/fake_machine_a.py
```

---

## 12. 推荐验收顺序

```text
1. 云端硬件、磁盘和 uv 勘测
2. 根 Python 3.11 环境
3. 在线 RL Python 3.10 环境
4. 43 项在线 RL 单测
5. 10 项根项目轻量测试
6. 完整根项目 pytest
7. debug_rlt（10 step）
8. debug_rlt_joint（10 step）
9. 真实数据单卡 20 step smoke
10. 单卡 100 / 1000 step
11. 两卡 FSDP 20 step smoke
12. 四卡正式 RLT 训练
13. Machine A 推理服务
14. fake Machine A + Machine B 联调
15. ManiSkill adapter 与在线 RL 仿真闭环
```

每一层通过后再进入下一层；失败时保留命令、完整日志、GPU 状态和 checkpoint 路径以便定位。
