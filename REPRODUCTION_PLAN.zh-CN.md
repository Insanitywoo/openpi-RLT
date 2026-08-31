# openpi-RLT 完整复现计划

> **状态：** 规划完成，等待按阶段人工执行。  
> **最后更新：** 2026-08-31
> **执行原则：** 本计划默认由学习者亲自逐步执行。助手应先解释每一步的目标、原理、命令、预期输出与常见故障，再等待用户确认结果；除非用户明确要求“自动执行某项工作”，否则不得自行安装依赖、修改配置、启动训练、下载大文件或运行长时间任务。

相关理论参考：[RLT_THEORY_REFERENCE.zh-CN.md](RLT_THEORY_REFERENCE.zh-CN.md)。  
中文项目说明：[README.zh-CN.md](README.zh-CN.md)。  
在线 RL 运行时中文说明：[rlt_online_rl/README.zh-CN.md](rlt_online_rl/README.zh-CN.md)。

---

## 1. 复现目标与验收边界

本轮以**当前 `openpi-RLT` 仓库的可运行链路**为主复现基线，而不是一开始就严格对齐原论文或真实以太网插接成功率。

### 第一轮目标

在没有真实机器人的条件下，在云端多 GPU 环境完成：

1. openpi 与 `rlt_online_rl` 环境可安装、可测试；
2. RL-token（阶段 1）训练、checkpoint 保存与恢复可运行；
3. Machine A 能提供冻结 VLA/RLT 的 `z_rl` 与 `ref_chunk`；
4. Machine B 的 Actor、Learner、Replay 服务可运行；
5. 使用 ManiSkill 增加一个保持兼容的仿真适配层；
6. 在仿真中跑通 warmup → replay → learner 更新 → actor snapshot 热加载的闭环；
7. 日志、配置、replay、checkpoint 都可保存并恢复；
8. 现有和新增的关键算法单元测试通过。

### 第一轮不承诺的内容

- 不把公开数据上的结果等同于真实 Ethernet insertion 结果；
- 不在第一轮直接接入 ROS/AgileX 或真实机器人；
- 不在第一轮要求仿真 Actor 的成功率一定超过基线 VLA；
- 不在第一轮宣称严格复现 RLT 论文数值结果。

### 第一轮验收标准

- 软件链路全通；
- 算法关键单测通过；
- 至少运行多个仿真 episode；
- replay 数量持续增长；
- Learner 的 `global_step`、Actor 的 version 持续增长；
- Actor snapshot 能被 Actor service 热加载；
- 重启服务后 checkpoint 和 replay journal 可以恢复。

---

## 2. 已确定的前提与资源规划

### 硬件与拓扑

- 本机 RTX 5070 Ti（约 16 GB 显存）：用于代码阅读、轻量测试、fake 服务和配置验证；不作为正式大模型训练主机。
- 本机 RTX 5070 Ti：优先完成源码理解、uv 环境、轻量测试和 fake/synthetic 验证；不要求承载完整 Pi0 模型。
- 百舸云 Pro6000 服务器：作为正式训练和仿真平台；本地验证通过后再同步代码和环境，按需创建开发机、分配 CPU、内存和 GPU。
- 初期采用**单机多进程/多 GPU**模拟 Machine A 与 Machine B；后续如有需要再拆分到多机局域网。

推荐 GPU 分工：

```text
GPU 0：Machine A，冻结 VLA/RLT 推理
GPU 1：Machine B Learner，Actor-Critic 训练
GPU 2+：ManiSkill 仿真、评估、数据预处理（按实际配额调整）
```

### 数据策略

- 没有论文原始数据；
- 阶段 1 优先使用公开的、格式接近的 LeRobot 数据集；
- 首轮可接受任务不完全等同于插接任务，目标是验证数据加载、归一化、RLT 训练与服务链路；
- 若公开数据适配未准备好，先使用 fake/synthetic 数据完成训练 smoke test；
- 之后才收集或接入接近真实目标任务的数据。

### 仿真策略

- 选择 **ManiSkill**；
- 保持现有 ROS/AgileX 真实机器人路径不变；
- 新增仿真适配层，复用现有 Actor、Critic、Learner、Replay、Machine A payload 与 `EnvDriver` 协调逻辑；
- 第一轮选择具备成功条件、稳定 reset、接触或对齐性质的任务，不强制一开始完全等同 Ethernet insertion。

---

## 3. 总体链路

```text
阶段 1：RLT 训练

公开/合成示范数据
  │
  ▼
pi0.5 VLA prefix embedding
  │
  ▼
RL-token Encoder / Decoder
  │
  ▼
RLT checkpoint


阶段 2：在线 RL

ManiSkill 观测
  │
  ▼
Machine A（冻结 VLA + RLT）
  ├── z_rl
  └── ref_chunk
          │
          ▼
Machine B Actor(z_rl, proprio, ref_chunk)
          │
          ▼
动作块执行到 ManiSkill
          │
          ▼
reward / done / next observation
          │
          ▼
Replay Manager → Learner → Actor snapshot → Actor service 热加载
```

---

## 4. 分阶段执行清单

每一阶段都必须满足：**先解释 → 用户执行 → 用户反馈结果 → 再进入下一步**。不要因为计划中存在命令就自动执行。

### 阶段 0：云端环境勘测与实验目录

**目标：** 确认云端开发机能承载正式训练与服务进程。

**用户需亲自执行的检查：**

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

**要记录的结果：**

- GPU 型号、数量、每卡显存；
- NVIDIA 驱动版本；
- CPU 核数、内存、数据盘可用空间；
- Ubuntu/内核版本；
- Python、`uv`、Conda、Docker（如使用）的版本；
- SSH、数据盘挂载点、可用端口范围。

**验收：** 可以确定训练、推理、仿真分别使用哪些 GPU，以及 checkpoint/replay/data 写入哪个持久化数据盘。

---

### 阶段 1：安装项目环境与运行已有测试

**目标：** 不训练、不下载大型模型之前，确认 Python/JAX/项目依赖可稳定工作。

根项目环境（Python 3.11）预期命令：

```bash
cd /path/to/openpi-RLT
uv sync --locked
```

在线 RL 子项目（Python 3.10）预期命令：

```bash
cd /path/to/openpi-RLT
uv venv --python 3.10 rlt_online_rl/.venv
uv pip install --python rlt_online_rl/.venv/bin/python \
  -e packages/openpi-client \
  -e 'rlt_online_rl[dev]'
```

**应先解释的重点：**

- 根项目与在线 RL 子项目要求不同 Python 版本；
- 根项目负责 VLA/RLT 训练和 Machine A；
- 子项目负责 Machine B、Replay 与在线 Actor-Critic；
- `jax[cuda12]` 与驱动/CUDA 兼容性必须先验证。

**用户执行的测试：**

```bash
cd /path/to/openpi-RLT
uv run pytest

cd /path/to/openpi-RLT/rlt_online_rl
python -m pytest tests
```

**验收：** 配置、网络、Replay、训练器相关单测通过；若失败，先定位版本或依赖问题，不进入训练阶段。

---

### 阶段 2：RLT 训练 smoke test

**目标：** 用小规模 fake/synthetic 数据验证 RLT 阶段 1 的计算图、loss、checkpoint 和恢复机制。

关键实现：

```text
src/openpi/models/rl_token.py
scripts/train_rlt.py
src/openpi/training/config.py
```

**需验证的逻辑：**

```text
VLA prefix embedding
  → RLTokenEncoder
  → RL tokens
  → RLTokenDecoder
  → prefix 重建 MSE
```

损失：

```text
L_RLT = MSE(reconstructed_prefix, stop_gradient(prefix))
L_total = L_RLT + alpha * L_VLA  # 当 alpha > 0 时
```

**smoke 配置原则：**

```text
batch size：1～2
训练步数：10～100
小 action horizon 或 tiny 配置（如仓库支持）
保存至少一个 checkpoint
启用明确的实验目录，如 runs/rlt_smoke
```

**验收：**

- loss 正常计算；
- 参数更新；
- checkpoint 生成；
- 新进程可以加载 checkpoint；
- `rlt_alpha=0` 时仅 RLT 模块训练；
- `rlt_alpha>0` 时联合损失正常。

**禁止跳过的检查：** checkpoint 中应包含 VLA 前缀树与 `rlt_module` 参数树；否则后续 Machine A 无法正确服务。

---

### 阶段 3：公开 LeRobot 数据适配与短训练

**目标：** 让阶段 1 走真实数据加载、变换、归一化和训练，而不要求先获得真实插接数据。

**数据筛选原则：**

1. 优先 LeRobot 格式；
2. 有图像、状态、动作和任务文本/可注入 prompt；
3. 动作空间尽量接近关节控制和 7 维动作；
4. 可接受任务不同，但必须记录差异；
5. 不匹配时通过新增适配配置或转换脚本解决，不破坏上游数据路径。

**必须逐项检查：**

```text
图像：相机字段、分辨率、mask
状态：proprio 维度、关节顺序、夹爪定义
动作：absolute / delta、action_dim、action_horizon
语言：prompt 是否存在或如何注入默认 prompt
归一化：norm stats 或 quantile q01/q99
```

**用户执行的最小检查：**

- 打印一个 dataset sample 的 keys、shape、dtype；
- 跑一个 dataloader batch；
- 验证转换后输入满足模型 shape；
- 先运行极短训练，再运行正式短训练。

**验收：** 真实公开数据可稳定读取，norm stats 一致，短训练可保存和恢复。

---

### 阶段 4：启动并验证 Machine A

**目标：** 把阶段 1 checkpoint 部署成冻结 VLA/RLT 服务。

预期命令：

```bash
cd /path/to/openpi-RLT
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir <checkpoint-dir> \
  --port 8000 \
  --shared-prefix-inference
```

**服务输入/输出契约：**

```text
输入：observation dict
输出：
  z_rl       # 紧凑 RLT 特征
  ref_chunk  # 冻结 VLA 参考动作块
```

**必须验证：**

- checkpoint 能加载；
- 单样本请求；
- batch 请求；
- `z_rl` 和 `ref_chunk` 的 shape；
- 断连和超时；
- `--shared-prefix-inference` 仅是延迟优化，不能改变输出语义。

**无真实 VLA 时：** 先用 `rlt_online_rl/launch/fake_machine_a.py` 验证网络与 payload；它不能作为模型效果验证。

---

### 阶段 5：启动并验证 Machine B

**目标：** 独立验证 Actor、Learner、Replay 三服务。

预期启动：

```bash
cd /path/to/openpi-RLT/rlt_online_rl
conda activate rlt_online_rl310
python launch/launch_machine_b.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml
```

**核心数据：**

```text
z_rl         [batch, z_dim]
proprio      [batch, proprio_dim]
ref_chunk    [batch, chunk_len, action_dim]
action_chunk [batch, chunk_len, action_dim]
rewards      [batch, chunk_len]
```

**当前 Ethernet 默认配置：**

```yaml
action_dim: 7
chunk_len: 10
z_dim: 2048
proprio_dim: 7
action_representation: delta_chunk
```

**必须验证：**

- Replay append/sample；
- Learner Critic update；
- Actor 按更新周期更新；
- target 网络 soft update；
- Actor snapshot 写入与热加载；
- learner checkpoint 保存/恢复；
- metrics/status JSON 生成。

首轮需要使用较小的 warmup 和训练步数；不要直接使用生产配置中 `warmup_post_collect_updates: 20000`。

---

### 阶段 6：实现 ManiSkill 仿真适配层

**目标：** 在不改变 ROS 路径的前提下，让已有在线 RL 运行时驱动 ManiSkill。

**新增范围：**

```text
simulation/
  ManiSkill 环境封装
  observation / proprio 转换
  action chunk 执行
  reward / done 转换
  reset 与 episode 管理

configs/tasks/
  仿真任务配置

tests/
  observation/action shape
  chunk 执行
  replay 写入
```

**接口原则：** 仿真环境应能被现有 `EnvDriver` 以环境对象方式使用：

```python
reset() -> observation
step(action) -> next_observation, reward, done, info
execute_chunk(action_chunk) -> execution result
```

**适配要求：**

- 固定长度 action chunk；
- 每步 reward 和 done；
- 稳定 reset；
- 明确 success 条件；
- 动作空间不为 7 维时允许增加映射层，但不得改变 RLT 网络公共输入输出；
- 仿真 observation 需可构造成 Machine A 需要的 observation dict，或在 fake Machine A 阶段提供等价 payload。

**验收：** 单独运行仿真 adapter 时，能完成 reset、执行一个 chunk、返回合法 reward/done/observation。

---

### 阶段 7：仿真端到端在线 RL

**目标：** 跑通完整闭环。

```text
ManiSkill observation
  → Machine A / fake Machine A
  → z_rl + ref_chunk
  → Actor
  → action chunk
  → ManiSkill execution
  → reward / done / next observation
  → Replay
  → Learner
  → Actor snapshot
  → Actor service reload
```

按三层推进：

1. **fake Machine A 软件闭环**：验证通信、payload、replay、learner、snapshot；
2. **冻结 reference baseline**：仅执行 `ref_chunk`，记录 return、success、长度、延迟；
3. **online Actor-Critic**：warmup 后允许 Actor 控制，并对照 baseline 日志。

**每轮必须记录：**

```text
experiment name / git commit / resolved config
GPU 分配 / 数据版本 / norm stats
episode return / success / length
replay size / learner global_step / actor version
Machine A 与 Actor 延迟
checkpoint / snapshot / journal 路径
```

**验收：** 多回合完成、replay 增长、learner 更新、actor 版本更新、snapshot 热加载与重启恢复均成立。

---

### 阶段 8：后续真实机器人扩展（暂不执行）

在仿真闭环稳定后，才开始：

1. 对齐 AgileX 的动作维度、关节顺序、相机、proprio 和 reset action；
2. 安装/验证 ROS2 Humble；
3. 检查 `train_deploy_alignment/` 中 ROS 适配器；
4. 做无运动 dry run；
5. 以低速、限位、急停和人工监督方式做 warmup；
6. 最后执行真实在线 RL。

任何真实机器人执行前必须单独制定安全清单；不得把仿真命令直接迁移到真实机器人。

---

## 5. 算法检查表

### RLT 阶段 1

```text
VLA prefix → RLTokenEncoder → RL tokens → RLTokenDecoder → 重建 MSE
```

- `rlt_alpha = 0`：只训练 RLT 模块；
- `rlt_alpha > 0`：`L_total = L_RLT + alpha * L_VLA`；
- RLT 重建目标上的 VLA embedding 必须 `stop_gradient`。

### 在线 RL 阶段 2

Actor 输入：

```text
z_rl + proprio + ref_chunk
```

Actor 损失：

```text
actor_loss = bc_weight * bc_penalty
           - q_weight * actor_q
           + delta_weight * delta_penalty
```

- `BASE/RL` 数据：BC target 是 `ref_chunk`；
- `HUMAN/MIXED` 数据：逐步使用实际 `action_chunk`；
- Critic 使用 Twin Q；
- TD target 使用 chunk 内折扣回报和 `gamma^H` bootstrap；
- reference dropout 只作用于 Actor 训练输入，不应改变部署输入或 BC target；
- `delta_chunk` 训练输出在执行前要反归一化并恢复为绝对动作。

---

## 6. 实验目录与可追溯性

所有云端实验都应写入持久化数据盘，使用如下布局：

```text
runs/<experiment_name>/
  checkpoints/
  actor_snapshot/
  replay/
  logs/
  metrics/
  configs/
  wandb/
  environment/
```

每个实验开始前保存：

- Git commit 与 `git diff`；
- 完整 resolved config；
- GPU/驱动/Python/JAX 版本；
- 数据集版本与挂载路径；
- norm stats 文件路径及校验信息；
- 启动命令和 GPU 分配；
- 目标、预期时长和停止条件。

---

## 7. 当前执行顺序

```text
1. 云端环境勘测
2. 安装根项目与 rlt_online_rl 环境
3. 运行全部现有单元测试
4. RLT fake/smoke 训练与 checkpoint 恢复
5. fake Machine A + Machine B 服务验证
6. 公开 LeRobot 数据检查与短训练
7. 真实 RLT checkpoint 的 Machine A 验证
8. ManiSkill adapter 实现与单测
9. fake Machine A + ManiSkill 端到端闭环
10. frozen reference baseline
11. online Actor-Critic 仿真实验
12. base/online 对照、日志归档
13. 再规划真实 AgileX/ROS 路线
```

---

## 8. 交互式学习执行约定

后续用户说“开始执行”“进行下一步”时，助手必须按以下固定格式响应：

1. **本步目标**：说明它在整条 RLT 链路中的位置；
2. **原理说明**：解释为何要做、相关软件/算法概念；
3. **执行前检查**：环境、目录、风险和前置条件；
4. **用户命令**：给出用户应亲自在终端执行的最小命令；
5. **预期输出**：说明成功时应看到什么；
6. **结果判读**：用户贴输出后，解释结果与故障排查；
7. **下一步门槛**：只有本步验收通过才进入下一步。

默认禁止助手自动执行以下操作：

- 安装、升级或卸载依赖；
- 下载模型、数据集或 Docker 镜像；
- 修改代码、配置或环境文件；
- 启动训练、服务、仿真或任何长时间进程；
- 远程登录、上传/下载、创建云端实例；
- 控制机器人或调用 ROS 动作接口。

只有当用户明确使用类似下述措辞时，助手才可代为执行对应范围内的动作：

```text
“请自动执行第 X 步”
“请帮我安装这些依赖”
“请直接修改该配置”
“请在当前环境运行此测试”
```

即使获得自动执行授权，对网络下载、云端操作、破坏性操作和真实机器人控制仍应在动作前重新确认范围与风险。
