# openpi-RLT 百舸云部署与训练指南

> **最后更新：** 2026-09-07
> **适用环境：** 当前通过 SSH 别名 `openpi-rlt` 访问的百舸云单 GPU 开发机。
> **当前路线：** 单卡基础环境 → 软件与算法测试 → fake RLT smoke → Pi0 + 公开 ALOHA → Machine A / Machine B → fake 环境 → ManiSkill。
> **执行原则：** 每一阶段先验收再进入下一阶段；长任务使用 `tmux`，长期资产写入 CFS；默认不使用 `--overwrite`。

相关文档：

- [长期复现计划](REPRODUCTION_PLAN.zh-CN.md)
- [RLT 理论参考](RLT_THEORY_REFERENCE.zh-CN.md)
- [在线 RL 运行时](rlt_online_rl/README.zh-CN.md)
- [本地复现报告](LOCAL_REPRODUCTION_REPORT.zh-CN.md)
- [本地复现教程](LOCAL_REPRODUCTION_TUTORIAL.zh-CN.md)

---

## 1. 当前目标与复现边界

首轮云端工作的目标不是直接复现真实 AgileX Ethernet insertion 成功率，而是完整观察并验收以下链路：

```text
云端双 Python 环境
→ 根项目与在线 RL 测试
→ fake-data RLT 阶段 1
→ 公开 Pi0 权重 + 公开 ALOHA 数据
→ Pi0 + ALOHA RLT checkpoint
→ Machine A：z_rl + ref_chunk
→ Machine B：Actor + Learner + Replay
→ 确定性 fake 环境
→ ManiSkill 仿真闭环
```

本轮不包含：

- 直接控制真实 AgileX 机器人或启动 ROS 物理执行器；
- 一开始进行多机或多 GPU 分布式训练；
- 将 fake 数据、fake 环境或 ManiSkill 结果等同于真实机器人效果；
- 在未验证数据字段、动作维度和归一化统计前直接训练 Pi0.5/AgileX 配置。

当前 AgileX 配置依赖三路相机、专用 state、32 维模型动作和私有数据约定。第一条公共可复现基线因此选择：

```text
基础模型：Pi0
数据：lerobot/aloha_sim_transfer_cube_human
已有 VLA 配置：pi0_aloha_sim
预训练权重：gs://openpi-assets/checkpoints/pi0_base/params
计划新增 RLT 配置：rlt_pi0_aloha、rlt_pi0_aloha_joint
```

截至 2026-09-07，`rlt_pi0_aloha` 和 `rlt_pi0_aloha_joint` **尚未加入代码**。在完成对应配置实现和测试之前，不应把它们写成已经可运行的训练入口。

---

## 2. 当前云端开发机快照

以下信息于 2026-09-07 通过 `ssh openpi-rlt` 实测：

```text
操作系统：Ubuntu 22.04.5 LTS
CPU：180 logical CPUs
内存：约 1.8 TiB
Swap：0

GPU 数量：1
nvidia-smi 设备名：NVIDIA RPBZZZ6
显存：97887 MiB，约 98 GiB
Compute Capability：12.0
驱动：580.159.04

系统盘：79 GiB，总体可用约 78 GiB
CFS：20 TiB，总体可用约 8.4 TiB

uv：0.10.12
系统 python3：3.12.13
git：未安装
tmux：未安装
```

项目目标资源记作百舸云 B300 单卡，但硬件记录和故障报告应保留 `nvidia-smi` 的实际设备字符串，不用项目称呼替代实测值。

远程仓库已经存在：

```text
/mnt/cfs/usr/wujh/openpi-RLT/repo/openpi-RLT
```

在未安装 `git` 的情况下，通过 `.git` 元数据确认：

```text
分支：main
提交：189a28de2bf90871fe89e92046766005326835b4
origin：https://github.com/Insanitywoo/openpi-RLT.git
```

已经创建：

```text
/root/workspace/openpi-rlt-system/env.sh
/root/workspace/openpi-rlt-system/openpi311/.venv
/root/workspace/openpi-rlt-system/online-rl310/.venv
```

uv 管理的 Python 解释器位于系统盘：

```text
/root/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/
/root/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/
```

当前状态是“云端 Git 快照、路径脚本和两个空的 Python 虚拟环境已准备；项目依赖、GPU Python 导入和训练尚未验收”，不是“已经可以开始训练”。

---

## 3. 系统盘与 CFS 分工

### 3.1 硬规则

系统盘 `/root/workspace` 只保存可重建的运行时：

```text
/root/workspace/openpi-rlt-system/
├── env.sh
├── openpi311/.venv/       # 根 openpi、RLT 阶段 1、Machine A
└── online-rl310/.venv/    # rlt_online_rl、Machine B
```

CFS 保存代码和长期资产：

```text
/mnt/cfs/usr/wujh/openpi-RLT/
├── repo/openpi-RLT/
├── datasets/
├── pretrained/
├── checkpoints/
│   ├── rlt_stage1/
│   └── online_rl/
├── runs/
├── replay/
├── cache/
│   ├── huggingface/
│   ├── openpi/
│   ├── torch/
│   ├── uv/
│   └── wandb/
├── configs/
├── logs/
├── reports/
└── artifacts/
```

禁止：

- 在 CFS 仓库中创建 `.venv`；
- 将数据、权重、checkpoint、replay 或长日志放在系统盘；
- 操作 `/mnt/cfs/usr` 下其他用户的目录；
- 把 SSH 私钥、Hugging Face token、W&B token 写入仓库。

JAX 编译缓存默认由当前训练代码写入 `~/.cache/jax`。它是可重建的临时编译产物，可以保留在系统盘；不要将“长期缓存必须在 CFS”误解为所有临时文件都必须迁移。

### 3.2 网络边界与本地资产传输策略

截至 2026-09-07，云端网络实测结果为：

```text
可访问：Ubuntu 百度镜像、PyPI、files.pythonhosted.org、Google Storage、Astral/uv
超时：GitHub HTTPS、Hugging Face HTTPS
```

因此不要把“云端可以访问 PyPI”误解为“云端可以访问所有依赖和数据源”。后续按以下分工执行：

```text
本地：代码修改、Git 提交、Git bundle、wheelhouse、源码归档、模型和数据准备
云端：创建本地 Python 环境、安装/导入依赖、GPU 验收、测试、训练、服务和实验记录
```

不同资产使用不同传输方式：

| 资产 | 推荐方式 | 说明 |
| --- | --- | --- |
| Git commit/branch/tag | Git bundle | 只传 Git 对象和历史，不包含普通未跟踪大文件 |
| 普通源码快照 | `scp` / `rsync` | 适合非 Git 文件或离线补丁 |
| Python wheel | 云端 PyPI 直装；缺失项才用本地 `wheelhouse` | 不默认打包完整环境；离线项用 `--no-index --find-links` |
| GitHub 依赖源码 | 云端 Git mirror 或本地源码归档 | 当前 `lerobot` mirror 已存在于 CFS |
| 模型权重和数据集 | `rsync`、`tar` 或 `tar.zst` | 直接放 CFS 的 `pretrained/`、`datasets/` |
| checkpoint/replay/日志 | 云端直接写 CFS | 不进入 Git，不放系统盘 |
| `.venv`/`site-packages` | 不传输 | 云端重新创建，避免绝对路径和 CUDA/编译兼容问题 |

Git bundle 只用于代码历史。不要把模型、数据、虚拟环境或 wheelhouse 塞进 Git bundle。

当前 CFS 已有 LeRobot mirror：

```text
/mnt/cfs/usr/wujh/openpi-RLT/cache/git-mirrors/lerobot.git
```

项目锁定的 LeRobot revision 已存在：

```text
0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

后续根环境安装采用混合策略：

1. PyPI 和 `files.pythonhosted.org` 可达，锁定的普通 wheel 在云端直接下载；
2. 默认依赖中唯一必须访问 GitHub 的包是 `lerobot`，优先重定向到上述 CFS mirror；
3. `dlimp` 也是 GitHub 依赖，但只属于可选的 `rlds` dependency group，首轮不安装；
4. 只有具体 wheel 在云端无法获取时，才在本地制作该包或该小组依赖的 wheelhouse；
5. 不默认打包全部 241 个包，因为 PyTorch/CUDA wheel 体积大，上传完整 wheelhouse 往往比云端直连 PyPI 更慢。

如果某个 Git 依赖仍无法解析，再由本地准备源码归档或 wheel。不要在没有网络诊断的情况下反复执行 `uv sync`。

### 3.3 环境解释与云端职责

系统自带 Python 是：

```text
/usr/bin/python3
/usr/bin/python3.12
```

它由 Ubuntu 系统管理，不作为本项目默认运行时。`uv python install 3.11` 和 `uv python install 3.10` 使用 uv 管理的 Python，当前安装在：

```text
/root/.local/share/uv/python/
```

`uv venv` 再在系统盘创建项目环境：

```text
/root/workspace/openpi-rlt-system/openpi311/.venv
/root/workspace/openpi-rlt-system/online-rl310/.venv
```

`uv` 的下载缓存通过 `UV_CACHE_DIR` 放在 CFS；Python 解释器和虚拟环境放在系统盘。`/root/.local/bin is not on PATH` 的警告不会影响后续使用绝对路径调用虚拟环境解释器。

项目依赖不应依赖系统 Python 的预装库。Ubuntu apt 主要负责系统库和工具，PyPI/源码/wheelhouse 负责 Python 项目依赖。云端必须重新创建虚拟环境并进行 GPU/CUDA/JAX/PyTorch 验收，但不上传本地 `.venv` 代替这一过程。

当前云端 apt 软件源包括：

```text
百度 Ubuntu 22.04 Jammy 镜像
NVIDIA CUDA Ubuntu 22.04 软件源
deadsnakes PPA
```

当前云端已有的相关工具包括：

```text
gcc、g++、make、cmake、ninja、rsync、tar、ffmpeg、nvcc、nvidia-smi、git、tmux、uv
```

`nvcc` 路径存在，但本轮只读检查没有获得有效版本输出；这不能作为 CUDA Toolkit 完整可用的证据。后续以 JAX/PyTorch 实际加载 GPU 和执行最小算子为准。

当前尚未发现但以后按实际错误再安装：

```text
pkg-config、git-lfs、wget、zstd
```

不要为了“可能需要”提前安装整套系统包或执行 `apt upgrade`。

### 3.4 环境变量脚本

在系统盘创建统一脚本：

```bash
mkdir -p /root/workspace/openpi-rlt-system

cat > /root/workspace/openpi-rlt-system/env.sh <<'EOF_ENV'
export OPENPI_CFS_ROOT=/mnt/cfs/usr/wujh/openpi-RLT
export OPENPI_REPO="$OPENPI_CFS_ROOT/repo/openpi-RLT"

export OPENPI311_VENV=/root/workspace/openpi-rlt-system/openpi311/.venv
export ONLINE_RL310_VENV=/root/workspace/openpi-rlt-system/online-rl310/.venv

export HF_HOME="$OPENPI_CFS_ROOT/cache/huggingface"
export OPENPI_DATA_HOME="$OPENPI_CFS_ROOT/cache/openpi"
export TORCH_HOME="$OPENPI_CFS_ROOT/cache/torch"
export XDG_CACHE_HOME="$OPENPI_CFS_ROOT/cache"
export UV_CACHE_DIR="$OPENPI_CFS_ROOT/cache/uv"
export WANDB_DIR="$OPENPI_CFS_ROOT/cache/wandb"
EOF_ENV

source /root/workspace/openpi-rlt-system/env.sh
```

`OPENPI_DATA_HOME` 是 OpenPI 下载器实际读取的缓存变量；只设置 `XDG_CACHE_HOME` 不会改变 OpenPI 默认的 `~/.cache/openpi`。

验证：

```bash
printf 'OPENPI_CFS_ROOT=%s\n' "$OPENPI_CFS_ROOT"
printf 'OPENPI_REPO=%s\n' "$OPENPI_REPO"
printf 'OPENPI311_VENV=%s\n' "$OPENPI311_VENV"
printf 'ONLINE_RL310_VENV=%s\n' "$ONLINE_RL310_VENV"
printf 'OPENPI_DATA_HOME=%s\n' "$OPENPI_DATA_HOME"
printf 'UV_CACHE_DIR=%s\n' "$UV_CACHE_DIR"
```

成功标准：所有路径分别指向约定的 CFS 或系统盘位置，没有出现空值和旧路径 `/mnt/openpi-rlt`。

---

## 4. 阶段 0：基础工具、目录和仓库验收

### 4.1 SSH 登录

从本机进入开发机：

```bash
ssh openpi-rlt
```

后续每次新 shell 先执行：

```bash
source /root/workspace/openpi-rlt-system/env.sh
```

首次尚未创建 `env.sh` 时，先完成第 3.4 节。

### 4.2 安装基础工具

当前系统是 Ubuntu 22.04，包管理器是 `apt-get`。安装会修改系统盘，执行前确认当前开发机允许安装系统包：

```bash
apt-get update
apt-get install -y git tmux
```

验证：

```bash
git --version
tmux -V
uv --version
```

成功标准：三个命令均正常返回版本。

### 4.3 检查硬件与挂载

```bash
date -Is
uname -a
nproc
free -h
df -h / /mnt/cfs
nvidia-smi
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,compute_cap --format=csv,noheader
python3 --version
uv --version
```

系统 `python3` 为 3.12 是正常的；项目不会直接使用它，而是由 uv 准备 Python 3.11 和 3.10。

### 4.4 检查远程仓库

```bash
source /root/workspace/openpi-rlt-system/env.sh

test -d "$OPENPI_REPO/.git"
git -C "$OPENPI_REPO" status --short --branch
git -C "$OPENPI_REPO" rev-parse HEAD
git -C "$OPENPI_REPO" remote -v
```

当前参考提交：

```text
189a28de2bf90871fe89e92046766005326835b4
```

如果本地已经 push 新提交，再在云端执行：

```bash
git -C "$OPENPI_REPO" pull --ff-only origin main
```

若远程目录缺失，才重新 clone：

```bash
mkdir -p "$OPENPI_CFS_ROOT/repo"
git clone https://github.com/Insanitywoo/openpi-RLT.git "$OPENPI_REPO"
```

成功标准：工作区无意外修改，分支为 `main`，远程提交与准备执行的本地提交一致。

---

## 5. 阶段 1：建立两个系统盘 uv 环境

根项目要求 Python `>=3.11`；`rlt_online_rl` 要求 Python `>=3.10,<3.11`。两套环境不能合并。

### 5.1 根 OpenPI/RLT 环境

安装 uv 管理的 Python 3.11，并在系统盘创建环境：

```bash
source /root/workspace/openpi-rlt-system/env.sh

uv python install 3.11
mkdir -p "$(dirname "$OPENPI311_VENV")"
uv venv --python 3.11 "$OPENPI311_VENV"
```

激活外部环境后，将根项目按锁文件同步进去：

```bash
source "$OPENPI311_VENV/bin/activate"
cd "$OPENPI_REPO"
uv sync --active --locked
```

这里必须使用 `--active`。否则 uv 可能在 CFS 仓库中创建项目 `.venv`，违反存储规则。

验证解释器位置：

```bash
"$OPENPI311_VENV/bin/python" - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY
```

成功标准：解释器位于 `/root/workspace/openpi-rlt-system/openpi311/.venv/`，版本为 Python 3.11。

### 5.2 在线 RL 环境

```bash
source /root/workspace/openpi-rlt-system/env.sh

uv python install 3.10
mkdir -p "$(dirname "$ONLINE_RL310_VENV")"
uv venv --python 3.10 "$ONLINE_RL310_VENV"

cd "$OPENPI_REPO"
uv pip install \
  --python "$ONLINE_RL310_VENV/bin/python" \
  -e packages/openpi-client \
  -e 'rlt_online_rl[dev]'
```

验证：

```bash
"$ONLINE_RL310_VENV/bin/python" - <<'PY'
import sys
import jax
import flax
import optax
import rlt_online_rl

print("Python:", sys.executable, sys.version)
print("JAX:", jax.__version__)
print("Flax:", flax.__version__)
print("Optax:", optax.__version__)
print("rlt_online_rl:", rlt_online_rl.__file__)
PY
```

成功标准：解释器位于 `/root/workspace/openpi-rlt-system/online-rl310/.venv/`，版本为 Python 3.10，所有包可导入。

### 5.3 检查仓库没有云端 `.venv`

```bash
find "$OPENPI_REPO" -maxdepth 2 -type d -name .venv -print
```

当前云端规范下应无输出。如果仓库本身从其他机器同步了 `.venv`，不要直接复用；虚拟环境不是可移植资产。

---

### 5.4 依赖安装前的资产准备关卡

两个虚拟环境已经创建，但当前不要直接运行会等待 GitHub 超时的 `uv sync`。依赖来源固定为：

1. 普通 Python wheel：云端从可访问的 PyPI/`files.pythonhosted.org` 按 `uv.lock` 下载；
2. `lerobot`：使用 CFS 中已有且包含锁定 revision 的 bare Git mirror；
3. `dlimp`：首轮不启用 `rlds` group，因此暂不处理；
4. Hugging Face 数据：本地下载后用 `rsync`/归档上传到 CFS，或后续使用平台对象存储；
5. OpenPI GCS 权重：优先由云端直接下载到 `$OPENPI_DATA_HOME`；
6. 只有明确失败的包才制作小范围 wheelhouse，不默认传输整个本地 Python 环境。

安装完成后仍必须在云端验证 Python、JAX、PyTorch、CUDA 和项目测试。

不要复制本地 `.venv`，不要把 wheelhouse、模型或数据写入 Git。默认资产暂存目录为：

```text
本地临时目录：/tmp/openpi-rlt-transfer/<commit>/
云端 CFS：/mnt/cfs/usr/wujh/openpi-RLT/artifacts/<commit>/
云端 wheelhouse：/mnt/cfs/usr/wujh/openpi-RLT/artifacts/python-wheelhouse/<commit>/
云端源码归档：/mnt/cfs/usr/wujh/openpi-RLT/artifacts/source/<commit>/
```

本关卡完成前，根环境和在线 RL 环境只保持空环境状态。

---

## 6. 阶段 2：导入、GPU 和测试验收

建议为每次验收创建独立日志目录：

```bash
source /root/workspace/openpi-rlt-system/env.sh
export CLOUD_TEST_RUN="cloud-test-$(date +%Y%m%d-%H%M%S)"
export CLOUD_TEST_LOG_DIR="$OPENPI_CFS_ROOT/logs/tests/$CLOUD_TEST_RUN"
mkdir -p "$CLOUD_TEST_LOG_DIR"
printf '%s\n' "$CLOUD_TEST_LOG_DIR"
```

### 6.1 根环境导入和设备发现

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$OPENPI311_VENV/bin/python" - <<'PY'
import sys
import jax
import flax
import torch
import openpi
import openpi_client

print("Python:", sys.executable, sys.version)
print("JAX:", jax.__version__)
print("JAX devices:", jax.devices())
print("Flax:", flax.__version__)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.cuda.is_available())
print("Torch GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("Torch GPU", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

成功标准：

- Python 来自根系统盘环境；
- JAX 至少返回一个 `CudaDevice`；
- PyTorch `torch.cuda.is_available()` 为 `True`；
- GPU 名称和 capability 被完整记录。

如果 PyTorch 只识别设备但运行 CUDA 算子报架构不兼容，应单独记录为 PyTorch wheel/Compute Capability 问题，不要与 JAX 是否可训练混为一谈。

### 6.2 在线 RL 单测

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"

"$ONLINE_RL310_VENV/bin/python" -m pytest -q rlt_online_rl/tests \
  2>&1 | tee "$CLOUD_TEST_LOG_DIR/rlt_online_rl_tests.log"
```

成功标准：无 failed、无 error。提交 `189a28d` 的参考结果是：

```text
43 passed
```

测试数量以后可能增加；应以“全部通过”为标准，而不是永久写死 43。

### 6.3 根项目轻量测试

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$OPENPI311_VENV/bin/python" -m pytest -q \
  src/openpi/transforms_test.py \
  scripts/test_rlt_batch.py \
  scripts/test_rlt_client.py \
  2>&1 | tee "$CLOUD_TEST_LOG_DIR/root_light_tests.log"
```

成功标准：无 failed、无 error。提交 `189a28d` 的参考结果是：

```text
10 passed
```

### 6.4 完整根项目测试

完整测试可能下载公开 OpenPI checkpoint，并触发大模型初始化和 JAX 编译。应在前两项通过后，使用 `tmux` 运行：

```bash
tmux new -s openpi-full-pytest
```

在 tmux 中：

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"

export FULL_TEST_LOG="$OPENPI_CFS_ROOT/logs/tests/full-pytest-$(date +%Y%m%d-%H%M%S).log"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
"$OPENPI311_VENV/bin/python" -m pytest -q \
  2>&1 | tee "$FULL_TEST_LOG"
```

常用 tmux 操作：

```text
分离：Ctrl-b d
恢复：tmux attach -t openpi-full-pytest
```

失败时保存完整 traceback、`nvidia-smi` 和日志路径。不要在未分类根因前反复删除环境或重装全部依赖。

---

## 7. 阶段 3：fake-data RLT smoke training

这一步验证：

```text
dummy Pi0.5 构建
→ VLA prefix
→ RLTokenEncoder / Decoder
→ reconstruction loss
→ alpha=0 或 alpha=1 的训练路径
→ 反向传播
→ checkpoint 保存与恢复
```

fake 数据只证明工程链路，不证明表示质量和任务效果。

### 7.1 冻结 VLA：`debug_rlt`

先使用唯一实验名，默认不覆盖已有目录：

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"

export RLT_SMOKE_EXP="cloud-debug-rlt-$(date +%Y%m%d-%H%M%S)"
export RLT_SMOKE_LOG="$OPENPI_CFS_ROOT/logs/training/$RLT_SMOKE_EXP.log"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$OPENPI311_VENV/bin/python" scripts/train_rlt.py debug_rlt \
  --checkpoint-base-dir "$OPENPI_CFS_ROOT/checkpoints/rlt_stage1" \
  --exp-name "$RLT_SMOKE_EXP" \
  --no-overwrite \
  --no-wandb-enabled \
  2>&1 | tee "$RLT_SMOKE_LOG"
```

`debug_rlt` 当前默认：

```text
fake data
batch size 2
10 train steps
1 个 RLT token
2 层 RLT
64 维 dummy embedding
rlt_alpha=0.0
```

验收目录：

```text
$OPENPI_CFS_ROOT/checkpoints/rlt_stage1/debug_rlt/$RLT_SMOKE_EXP/
```

10-step 运行的最终 checkpoint step 目录通常是 `9/`。检查：

```bash
find "$OPENPI_CFS_ROOT/checkpoints/rlt_stage1/debug_rlt/$RLT_SMOKE_EXP" \
  -maxdepth 3 -type f -o -type d | sort
```

### 7.2 checkpoint 恢复

保留上一节的同一个 `RLT_SMOKE_EXP`，把总步数从 10 提高到 12：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$OPENPI311_VENV/bin/python" scripts/train_rlt.py debug_rlt \
  --checkpoint-base-dir "$OPENPI_CFS_ROOT/checkpoints/rlt_stage1" \
  --exp-name "$RLT_SMOKE_EXP" \
  --resume \
  --no-overwrite \
  --num-train-steps 12 \
  --no-wandb-enabled \
  2>&1 | tee -a "$RLT_SMOKE_LOG"
```

成功标准：日志明确恢复已有状态，并继续到新增 step，而不是从 0 重新开始。

### 7.3 联合损失：`debug_rlt_joint`

```bash
export RLT_JOINT_EXP="cloud-debug-rlt-joint-$(date +%Y%m%d-%H%M%S)"
export RLT_JOINT_LOG="$OPENPI_CFS_ROOT/logs/training/$RLT_JOINT_EXP.log"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$OPENPI311_VENV/bin/python" scripts/train_rlt.py debug_rlt_joint \
  --checkpoint-base-dir "$OPENPI_CFS_ROOT/checkpoints/rlt_stage1" \
  --exp-name "$RLT_JOINT_EXP" \
  --no-overwrite \
  --no-wandb-enabled \
  2>&1 | tee "$RLT_JOINT_LOG"
```

`debug_rlt_joint` 使用 `rlt_alpha=1.0`，总损失包含 RLT 重建项和 VLA loss。验收时应看到 `loss`、`rlt_loss`、`mse` 和 `vla_loss`，并确认 checkpoint 可保存。

只有明确决定废弃同名实验目录时才使用 `--overwrite`。删除 checkpoint、覆盖实验或清理日志前应再次确认影响。

---

## 8. 阶段 4：公开 ALOHA 数据和 Pi0 权重验收

已有公共 VLA 配置：

```text
配置：pi0_aloha_sim
数据：lerobot/aloha_sim_transfer_cube_human
权重：gs://openpi-assets/checkpoints/pi0_base/params
默认 prompt：Transfer cube
```

在新增 RLT 配置前，先验收：

- Hugging Face 数据下载进入 `$HF_HOME`；
- OpenPI/GCS 权重下载进入 `$OPENPI_DATA_HOME`；
- sample keys、dtype、shape 和相机字段符合 `LeRobotAlohaDataConfig`；
- state/action 维度与 `AlohaInputs`、`AlohaOutputs` 一致；
- action padding 和坐标转换可解释；
- norm stats 和 assets 可被定位；
- 单个 batch 可构建，Pi0 基础权重可加载。

可先运行相关数据与 policy 测试，但应注意 policy 测试可能下载额外 checkpoint：

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$OPENPI311_VENV/bin/python" -m pytest -q \
  src/openpi/training/data_loader_test.py \
  src/openpi/policies/policy_test.py
```

首次下载前检查：

```bash
printf 'HF_HOME=%s\n' "$HF_HOME"
printf 'OPENPI_DATA_HOME=%s\n' "$OPENPI_DATA_HOME"
du -sh "$HF_HOME" "$OPENPI_DATA_HOME" 2>/dev/null || true
```

验收失败时先判断是网络、认证、缓存路径、数据 schema、norm stats、权重结构还是 GPU 显存问题，不要直接进入 RLT 正式训练。

---

## 9. 阶段 5：实现 Pi0 + ALOHA RLT 配置

当前代码尚无：

```text
rlt_pi0_aloha
rlt_pi0_aloha_joint
```

下一阶段需要在不破坏 `pi0_aloha_sim` 的前提下增加：

```text
rlt_pi0_aloha：rlt_alpha=0，只训练 RLT Encoder/Decoder
rlt_pi0_aloha_joint：rlt_alpha>0，验证联合 VLA loss 路径
```

配置应复用 `pi0_aloha_sim` 的：

- Pi0 模型和动作规格；
- `LeRobotAlohaDataConfig`；
- ALOHA prompt 与输入输出变换；
- `pi0_base` 权重来源；
- 数据 assets 和归一化约定。

配置实现后的代码验收至少包括：

```text
配置名可通过 CLI 解析
数据 batch 可构建
RLT input_dim 与 Pi0 prefix hidden size 一致
alpha=0 时 VLA 参数保持冻结
alpha>0 时出现 vla_loss
20-step checkpoint 可保存和恢复
```

还必须区分“阶段 1 训练接口”和“部署接口”。当前代码的接口事实是：

```text
AlohaOutputs：输出 14 维双臂动作
serve_rlt_policy.py：固定 proprio_dim=7、action_dim=7、chunk_len=50
在线 RL AgileX 配置：action_dim=7、chunk_len=10
```

因此，Pi0 + ALOHA 可以先用于 RLT 阶段 1 的公共训练基线，但不能在未适配的情况下直接宣称已经接通现有 Machine A/B。进入部署阶段前，需要完成以下之一，并在代码和配置中固定选择：

1. 将 Machine A 输出 contract 参数化，由任务 adapter 明确产生 `z_rl`、`proprio` 和目标环境所需的 `ref_chunk`；
2. 新增独立的 ALOHA/ManiSkill 部署 adapter，将 14 维 ALOHA 参考动作转换为选定仿真任务的动作空间；
3. 如果只验证服务加载和 RLT 特征，则明确标记为“Machine A 模型加载 smoke”，不把截断后的前 7 维动作当作有效在线 RL reference。

在这两个训练配置和部署 contract 真正提交并同步到云端前，不执行后续真实 Machine A/B 联调命令。

---

## 10. 阶段 6：Pi0 + ALOHA RLT 训练阶梯

配置实现后，按下列阶梯逐级运行：

```text
20 steps → 100 steps → 1,000 steps → 5,000 steps
```

每一级都使用新的实验名，并检查：

- 数据加载和首 batch；
- 首次 JIT 编译时间；
- `loss`、`rlt_loss`、`mse`，联合版还包括 `vla_loss`；
- GPU 峰值显存和利用率；
- checkpoint 保存和恢复；
- 日志、权重和缓存全部位于约定目录。

配置完成后的 20-step 命令模板：

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"

export RLT_ALOHA_EXP="rlt-pi0-aloha-smoke20-$(date +%Y%m%d-%H%M%S)"
export RLT_ALOHA_LOG="$OPENPI_CFS_ROOT/logs/training/$RLT_ALOHA_EXP.log"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
"$OPENPI311_VENV/bin/python" scripts/train_rlt.py rlt_pi0_aloha \
  --checkpoint-base-dir "$OPENPI_CFS_ROOT/checkpoints/rlt_stage1" \
  --exp-name "$RLT_ALOHA_EXP" \
  --num-train-steps 20 \
  --batch-size 1 \
  --num-workers 2 \
  --no-overwrite \
  --no-wandb-enabled \
  2>&1 | tee "$RLT_ALOHA_LOG"
```

20-step smoke 通过后再逐级提高训练步数和 batch size。当前云端只有一张 GPU，第一轮不设置 `--fsdp-devices 2` 或四卡拓扑。只有单卡基线稳定且有明确显存/吞吐需求时，才规划多 GPU 资源。

---

## 11. 阶段 7：Machine A 与部署 contract 验收

Pi0 + ALOHA RLT checkpoint 完成后，先解决训练数据动作空间、服务输出和在线 RL 输入之间的 contract，不直接启动完整联调。

### 11.1 当前接口差异

截至提交 `189a28d`：

```text
AlohaOutputs：返回 [T, 14] 双臂动作
serve_rlt_policy.py：
  PROPRIO_DIM = 7
  CHUNK_LEN = 50
  ACTION_DIM = 7
  ref_chunk = actions[:50, :7]
AgileX online RL：期望 [10, 7] ref_chunk
```

在线 RL 的 feature coercion 会把足够大的 Machine A 输出裁剪到配置的 `[chunk_len, action_dim]`，所以 `[50, 7] → [10, 7]` 在 shape 上可通过；但 ALOHA 14 维动作被取前 7 维是否具有正确任务语义，必须由部署 adapter 明确，不能仅以 shape 可兼容作为验收。

Machine A 的目标 payload 仍是：

```python
{
    "z_rl": z_rl,
    "proprio": proprio,
    "ref_chunk": ref_chunk,
}
```

但 `proprio` 和 `ref_chunk` 的维度、单位、关节顺序、归一化状态和任务语义必须与目标环境配置一致。

### 11.2 服务监听安全前置条件

当前 `scripts/serve_rlt_policy.py` 将 WebSocket host 硬编码为：

```python
host="0.0.0.0"
```

CLI 目前只有 `--port`，没有 `--host`。这与项目“初期只监听 `127.0.0.1`”的约定不一致。正式在云端启动前，必须满足至少一项：

- 修改服务入口，增加 `--host`，默认或显式设置为 `127.0.0.1`；
- 在百舸云安全组/容器网络层确认端口 8000 不可被公网访问，并仅通过本机回环或 SSH tunnel 使用。

推荐后续代码修改采用第一种方式；在修改完成前，不把服务已安全绑定到本机写入实验报告。

### 11.3 分层验收

部署 adapter 完成后，先做模型加载 smoke，再做有效动作 contract 验收。命令模板为：

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
"$OPENPI311_VENV/bin/python" scripts/serve_rlt_policy.py \
  --config rlt_pi0_aloha \
  --checkpoint-dir "$OPENPI_CFS_ROOT/checkpoints/rlt_stage1/rlt_pi0_aloha/<exp-name>/<step>" \
  --port 8000 \
  --shared-prefix-inference
```

只有在增加 `--host` 后，才在命令中补充：

```bash
--host 127.0.0.1
```

`--shared-prefix-inference` 只复用 VLA prefix/KV cache，降低推理延迟；不改变模型权重、checkpoint、归一化或 online-RL payload。

验收分为两层：

1. **模型加载 smoke**：checkpoint、assets、单请求、重复请求、batch 请求和 RLT shape 正常；
2. **部署 contract**：`proprio`、`ref_chunk` 的 shape、dtype、单位、关节顺序、动作语义和在线 RL 配置全部一致。

同时记录启动后常驻显存、首请求编译峰值、缓存命中后的推理延迟、超时和断连行为。只有两层都通过，才能进入真实 Machine A + Machine B 联调。

---

## 12. 阶段 8：Machine B 与 fake 闭环

Machine B 使用 Python 3.10 环境，由以下进程组成：

```text
ReplayManager
LearnerService
ActorService
可选 W&B monitor
```

真实任务配置位于：

```text
rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml
```

不要直接修改仓库内任务配置。将实验副本写入 CFS：

```text
$OPENPI_CFS_ROOT/configs/online_rl/<run-name>.yaml
```

配置中的长期路径必须改到 CFS，例如：

```yaml
runtime:
  monitoring:
    wandb_dir: /mnt/cfs/usr/wujh/openpi-RLT/cache/wandb

  actor_service:
    snapshot_path: /mnt/cfs/usr/wujh/openpi-RLT/runs/online_rl/<run-name>/actor_snapshot/actor_snapshot.pkl

  learner_service:
    checkpoint_dir: /mnt/cfs/usr/wujh/openpi-RLT/runs/online_rl/<run-name>/checkpoints
    actor_snapshot_path: /mnt/cfs/usr/wujh/openpi-RLT/runs/online_rl/<run-name>/actor_snapshot/actor_snapshot.pkl

  replay:
    journal_path: /mnt/cfs/usr/wujh/openpi-RLT/replay/<run-name>/replay_journal.pkl

  env_driver:
    machine_a_ws_url: ws://127.0.0.1:8000
```

第一轮不要直接使用生产预算：

```text
warmup_min_size: 600
warmup_post_collect_updates: 20000
```

应先创建小预算实验副本，验证 RPC、Replay、checkpoint 和 snapshot，再逐步恢复正式值。

Machine B 启动模板：

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO/rlt_online_rl"

CUDA_VISIBLE_DEVICES=0 \
"$ONLINE_RL310_VENV/bin/python" launch/launch_machine_b.py \
  --config "$OPENPI_CFS_ROOT/configs/online_rl/<run-name>.yaml"
```

当前只有一张 GPU，不应预先假设 Machine A、Actor 和 Learner 可以按 `0.70 + 0.10 + 0.75` 的内存比例同时常驻。正确顺序：

```text
单独测量 Machine A
→ 使用 fake Machine A 单独测量 Machine B
→ 记录各进程常驻和峰值显存
→ 再决定同卡并发、串行运行或申请第二张 GPU
```

fake 验收顺序：

```text
在线 RL 单测
→ fake Machine A
→ 确定性 DummyChunkEnv
→ Replay 增长
→ Learner global_step 增长
→ actor snapshot 生成
→ Actor version 热更新
→ 服务重启恢复
```

真实 AgileX rollout 依赖 ROS 和机器人接口。在 ManiSkill adapter 完成前，不启动真实机器人 rollout。

---

## 13. 阶段 9：ManiSkill 与后续扩展

ManiSkill adapter 应作为独立仿真适配层，不破坏 ROS/AgileX 路径。最小接口：

```python
reset() -> observation
step(action) -> next_observation, reward, terminated, truncated, info
execute_chunk(action_chunk) -> execution_result
```

接入顺序：

```text
adapter 单测
→ reference-only baseline
→ fake Machine A + ManiSkill
→ 真实 Machine A + ManiSkill
→ warmup / replay / learner / snapshot
→ evaluation
```

至少记录：

```text
episode reward
success rate
episode length
action delta magnitude
BC penalty
Q value
actor loss
critic loss
fallback 次数
```

只有单卡训练、Machine A、Machine B 和 ManiSkill 闭环稳定后，才考虑：

- 多 GPU/FSDP；
- 将 Machine A 与 Learner 分卡；
- Docker 固化环境；
- Pi0.5/AgileX 专有数据；
- ROS2 Humble 和真实机器人。

---

## 14. 推荐验收顺序与当前进度

### 14.1 当前已完成

```text
[完成] 本机双环境和轻量软件验证
[完成] SSH 别名 openpi-rlt
[完成] 云端 GPU、CPU、内存、系统盘和 CFS 勘测
[完成] CFS 个人目录架构
[完成] 云端仓库存在，提交为 189a28d
[完成] 云端 uv 可用
```

### 14.2 当前下一步

```text
[待执行] 安装 git 和 tmux
[完成] 创建 /root/workspace/openpi-rlt-system/env.sh
[完成] 安装 uv 管理的 Python 3.11.15 和 3.10.20
[完成] 创建系统盘 Python 3.11 根环境
[完成] 创建系统盘 Python 3.10 在线 RL 环境
[待执行] 本地依赖 wheelhouse/源码资产准备
[待执行] 云端依赖安装与 Git mirror 配置
[待执行] 云端导入、GPU 与测试验收
```

### 14.3 后续阶段

```text
1. debug_rlt 及恢复测试
2. debug_rlt_joint
3. 公开 ALOHA 数据和 Pi0 权重验收
4. 实现 rlt_pi0_aloha / rlt_pi0_aloha_joint
5. 20-step Pi0 + ALOHA RLT smoke
6. 100 / 1,000 / 5,000-step 训练
7. Machine A 单独验收
8. fake Machine A + Machine B
9. 确定性 fake 环境闭环
10. ManiSkill adapter 与在线 RL 仿真闭环
11. 按实测需求规划多 GPU、Pi0.5/AgileX 和真实机器人
```

每一阶段通过后再进入下一阶段。失败时保留：

```text
Git commit
完整命令
环境变量摘要
Python/JAX/PyTorch/uv 版本
nvidia-smi
完整日志路径
数据/权重/checkpoint 路径
错误 traceback
```

故障应先分类为环境、网络、依赖、CUDA/JAX、显存、数据 schema、checkpoint、RPC 或算法数值问题，再决定下一步，不以“重装全部依赖”作为默认排障方式。

---

## 15. 2026-09-08：RLT 阶段 1 已验证运行时（Pro6000 / sm_120）

本节记录已在当前百舸云开发机实际验证的运行时，优先级高于本文中较早的“直接 `uv sync`”示例。它用于 RLT 阶段 1；在线 RL 的 Python 3.10 环境仍是下一阶段的独立工作。

### 15.1 已验证版本与兼容性结论

云端 GPU 的 compute capability 为 `sm_120`。项目初始锁定的 `torch==2.7.1+cu126` 虽能发现 CUDA 设备，但不包含 `sm_120` 内核；实际 CUDA 矩阵乘法会以 `no kernel image is available` 失败。因此不能将“`torch.cuda.is_available()` 为真”视为 GPU 可训练的验收。

当前根 RLT 环境已验证使用：

```text
torch==2.10.0+cu128
torchvision==0.25.0+cu128
Python 3.11.15
JAX 0.5.3
```

`torch==2.10.1+cu128` 在准备时没有可用的官方 wheel，因此使用与 `torchvision==0.25.0+cu128` 配对、且实际存在的 `torch==2.10.0+cu128`。该 wheel 的 arch list 包含 `sm_120`，并已在云端对 CUDA 矩阵乘法完成真实验收。

根项目的 `pyproject.toml` 和 `uv.lock` 已记录上述 PyTorch CUDA 12.8 来源。不要自行将它降回旧版，也不要混装不匹配的 `torch` / `torchvision` wheel。

### 15.2 JAX 私有 CUDA 运行时与统一启动包装器

JAX 0.5.3 与 PyTorch CUDA 12.8 用户态库直接混用时，JAX 可发现 GPU，但矩阵乘法可能失败。为保持锁定的 JAX 版本，已在**系统盘**保存它自己的私有 CUDA 动态库：

```text
/root/workspace/openpi-rlt-system/jax053-cuda-libs/nvidia
```

必须通过下列包装器启动所有根项目的 JAX/RLT 命令：

```text
/root/workspace/openpi-rlt-system/run-openpi-jax.sh
```

包装器会加载 `env.sh`，将该私有库目录置入 `LD_LIBRARY_PATH`，然后 `exec` 原命令。它不替换项目代码、不修改 CFS，也不适用于 Python 3.10 的在线 RL 环境。

示例（在云端 shell 内执行）：

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"
/root/workspace/openpi-rlt-system/run-openpi-jax.sh \
  "$OPENPI311_VENV/bin/python" scripts/train_rlt.py debug_rlt
```

最小运行时验收应同时包含 JAX 和 PyTorch 的**真实** CUDA 算子，而非仅打印版本或设备名称：

```bash
source /root/workspace/openpi-rlt-system/env.sh
/root/workspace/openpi-rlt-system/run-openpi-jax.sh \
  "$OPENPI311_VENV/bin/python" - <<'PY'
import jax
import jax.numpy as jnp
import torch

print("JAX devices:", jax.devices())
print("JAX matmul:", float((jnp.ones((32, 32)) @ jnp.ones((32, 32))).block_until_ready()[0, 0]))
print("Torch:", torch.__version__)
print("Torch arch:", torch.cuda.get_arch_list())
print("Torch matmul:", float((torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda"))[0, 0]))
PY
```

成功条件：JAX 返回 `CudaDevice`、两次 matmul 都有数值输出、PyTorch arch list 包含 `sm_120`。

### 15.3 离线 wheelhouse 与重建规则

已上传的 Python 3.11 CUDA 12.8 wheelhouse 位于 CFS：

```text
/mnt/cfs/usr/wujh/openpi-RLT/cache/wheelhouse/torch-2.10.0-cu128-cp311-linux-x86_64
```

其中包含 PyTorch、Torchvision、Triton 与匹配的 NVIDIA CUDA 用户态 wheel。它是可复用的重建资产，不进入 Git，也不应复制整个本地 `.venv`。

当前已验证环境上**不要直接执行**普通的：

```bash
uv sync --active --locked
```

原因是该命令可能重新解析/安装 CUDA 运行时，并破坏已经验证的 JAX/PyTorch 组合；仅 `source env.sh` 也不会激活 `$OPENPI311_VENV`，有创建仓库 `.venv` 的风险。

若系统盘环境因重建而丢失，应按“创建 Python 3.11 venv → 从上述 wheelhouse 离线安装 PyTorch CUDA 12.8 运行时 → 安装其余锁定依赖/本地 Git mirror → 恢复 JAX 私有运行时 → 用包装器执行双 matmul 验收”的顺序恢复，并把完整命令和日志记录到 CFS。不要在未确认网络与 wheel 来源时反复重试在线安装。

### 15.4 RLT 阶段 1 已完成验收

截至 **2026-09-08**，以下云端实测均已完成，日志与 checkpoint 保留在 CFS：

- `torch 2.10.0+cu128` 的 `sm_120` CUDA matmul；
- JAX 0.5.3 GPU matmul（经 `run-openpi-jax.sh`）；
- `openpi`、`openpi_client`、`lerobot` 与 RLT 模块导入；
- `debug_rlt` fake-data 10-step 训练及 checkpoint 保存；
- 同一实验从 checkpoint 恢复并续跑；
- `debug_rlt_joint` fake-data 10-step 联合损失训练，确认 `rlt_loss` 和 `vla_loss` 都产生；
- CFS checkpoint 写入。

对应训练日志：

```text
/mnt/cfs/usr/wujh/openpi-RLT/logs/training/cloud-debug-rlt-20260908-124731.log
/mnt/cfs/usr/wujh/openpi-RLT/logs/training/cloud-debug-rlt-joint-20260908-125330.log
```

这表示 RLT 阶段 1 的**软件、GPU、训练、恢复与持久化链路**已就绪；尚未下载真实 ALOHA/Hugging Face 数据或 Pi0 权重，也尚未开始正式训练。下一独立阶段是 Python 3.10 `rlt_online_rl` 环境和在线 RL 单测。

## 16. 2026-09-08：Online RL Python 3.10 环境已验证

Online RL 使用独立环境，不与根 OpenPI/RLT 的 Python 3.11 环境混用：

```text
解释器：/root/workspace/openpi-rlt-system/online-rl310/.venv/bin/python
Python：3.10.20
JAX：0.5.3
Flax：0.10.2
Optax：0.2.8
NumPy：1.26.4
OpenCV：4.11.0
```

由于云端直接下载 JAX CUDA 依赖长时间停滞，本次使用本地已验证的 Python 3.10 `site-packages` 离线归档传输到 CFS；CUDA vendor 动态库不重复传输，复用系统盘上的 JAX 私有运行时：

```text
归档：/mnt/cfs/usr/wujh/openpi-RLT/tmp/online-rl310-site-packages.tar.gz
JAX CUDA：/root/workspace/openpi-rlt-system/jax053-cuda-libs/nvidia
```

Online RL 命令统一使用：

```text
/root/workspace/openpi-rlt-system/run-online-rl.sh
```

该包装器加载 JAX 私有 CUDA 动态库后执行传入命令。例如：

```bash
source /root/workspace/openpi-rlt-system/env.sh
cd "$OPENPI_REPO"
/root/workspace/openpi-rlt-system/run-online-rl.sh \
  "$ONLINE_RL310_VENV/bin/python" -m pytest -q rlt_online_rl/tests
```

2026 年 9 月 8 日已完成：

```text
JAX GPU matmul：通过
rlt_online_rl 全部单测：43 passed
退出码：0
```

测试日志：

```text
/mnt/cfs/usr/wujh/openpi-RLT/logs/tests/online-rl310-20260908.log
```

初次验收时 pytest 曾提示根项目中的 `exclude-dependencies` 为未知配置项；2026-09-08 复核已移除该无效 pytest 配置，随后本地复测为无警告的 `43 passed`。Online RL 环境已具备单测和后续 fake Machine B 开发所需的 Python 依赖，但尚未启动在线 RL、Machine A/B、机器人、ROS 或正式训练。

## 17. 2026-09-08：双环境与项目配置复核

本节是完成根 RLT 与 Online RL 环境后进行的复核结论。

### 17.1 当前配置状态

| 项目 | 状态 | 结论 |
|---|---|---|
| 云端代码目录 | `/mnt/cfs/usr/wujh/openpi-RLT/repo/openpi-RLT` | 位于 CFS，符合长期资产规则 |
| 根 RLT 环境 | Python 3.11.15 | `openpi`、`openpi_client`、JAX 和 PyTorch 可导入 |
| 根 RLT GPU | JAX 0.5.3、PyTorch 2.10.0+cu128 | 两种框架均已执行实际 CUDA matmul；PyTorch 包含 `sm_120` |
| Online RL 环境 | Python 3.10.20 | `rlt_online_rl`、`openpi_client`、JAX、Flax、Optax、OpenCV 可导入 |
| Online RL GPU | JAX 0.5.3 | 经 `run-online-rl.sh` 实际 GPU matmul 通过 |
| RLT fake training | `debug_rlt`、恢复、`debug_rlt_joint` | 均完成；日志/checkpoint 在 CFS |
| Online RL 测试 | `rlt_online_rl/tests` | `43 passed`，复核后无 pytest 配置警告 |
| 网络安全配置 | fake/online 配置 | Actor、Replay、Machine A 默认都使用 `127.0.0.1`，未默认暴露公网端口 |

复核时系统盘可用空间约 `64G`，CFS 可用空间约 `8.3T`；当前环境和已有日志/checkpoint 不存在空间风险。

### 17.2 Online RL 为什么没有 PyTorch

Online RL 的 Actor、Critic、Learner 和网络实现直接使用 JAX/Flax/Optax；其项目依赖是 `jax[cuda12]==0.5.3`，没有 `torch` 或 `torchvision`，代码也没有直接导入 PyTorch。因此 Python 3.10 Online RL 环境**不需要安装 PyTorch**。

它仍然需要 CUDA：JAX 的 CUDA plugin、PJRT 和 NVIDIA 用户态动态库负责将 Actor/Critic 的矩阵计算和梯度更新放到 GPU。当前包装器加载 JAX 的私有 CUDA 动态库，且已执行真实 JAX GPU matmul；所以 Online RL 不是 CPU-only 环境。

根 RLT 环境必须额外安装 PyTorch，是因为 OpenPI 的数据/模型链路会导入它；该环境的 GPU 是 `sm_120`，旧 `torch==2.7.1+cu126` 缺少该架构内核，才需要升级到 `torch==2.10.0+cu128`。Online RL 不调用 PyTorch 内核，故不会重现“旧 PyTorch wheel 不支持 `sm_120`”的问题。JAX 也不是凭版本假设兼容：已在这张 `sm_120` GPU 上完成实际 matmul 验收。

### 17.3 Online RL 的锁文件与 ROS 边界

`rlt_online_rl/uv.lock` 已加入版本控制，Online RL 的普通 Python 依赖现在可被确定性解析。此前 `rlt_online_rl/pyproject.toml` 将 `rclpy` 放在可选 PyPI extra 中，导致 `uv lock` 在所有平台解析时失败；`rclpy` 是 ROS 2 系统分发包，不在 PyPI，不能作为 uv 可解析依赖。

因此项目元数据不再把 `rclpy` 声明为 uv extra。需要 ROS 的真实机器人脚本仍保留在仓库中，但它们属于后续 ROS 2 Humble 专用环境：届时先安装/加载系统 ROS，再建立与 ROS Python ABI 匹配的运行环境，不能在当前 fake/Online RL venv 中直接执行。

### 17.4 仍需遵守的运行规则

- 根 RLT/JAX 命令：`run-openpi-jax.sh`；
- Online RL/JAX 命令：`run-online-rl.sh`；
- 两个 venv 都在系统盘，代码、日志、数据、wheelhouse、模型和 checkpoint 都在 CFS；
- 不在已验收根环境随意执行普通 `uv sync --active --locked`；恢复时使用记录的离线 wheelhouse/归档和双 matmul 验收；
- 云端 GitHub HTTPS 仍不可达。CFS 云端仓库与本地采用 Git bundle 同步；本地 `origin` 是否已 push 到 GitHub 需单独以 `git status --branch` 和 `git ls-remote` 确认，不能把云端的本地 `origin/main` 跟踪引用误认为 GitHub 已同步。
