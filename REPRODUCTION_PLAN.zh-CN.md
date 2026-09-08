# openpi-RLT 云端完整复现计划

> **版本：** v2（Pi0 + ALOHA + fake 环境 + ManiSkill）
> **状态：** 已确定路线，按阶段交互式执行
> **最后更新：** 2026-09-08
> **执行原则：** 助手每次只讲一个小步骤；先解释目标、原理、命令、预期输出和故障，再由用户亲自执行并反馈。除非用户明确要求自动执行，否则不自动安装依赖、下载大文件、修改训练配置或启动长任务。

## 1. 本轮目标与边界

本轮目标是在百舸云 B300 单卡开发机上完整观察一条可运行链路：

```text
公开 Pi0 权重 + 公开 ALOHA 数据
→ RLT 阶段一训练
→ RLT checkpoint
→ Machine A（VLA/RLT 特征与参考动作服务）
→ fake Machine A / fake 环境验证在线 RL 基础设施
→ Machine B（Actor、Learner、Replay）
→ ManiSkill 仿真适配
→ warmup → online → snapshot → evaluation
```

本轮不包含：

- 真实 AgileX 机器人、ROS2 和 Ethernet insertion；
- 直接声称复现论文或真实机器人成功率；
- 一开始进行多机分布式训练；
- 把 fake 数据或 fake 环境结果当成算法效果。

第一轮验收是软件链路、算法逻辑、数据接口、checkpoint 恢复和仿真闭环全部可追踪。

## 2. 已确定的技术路线

### 2.1 为什么需要两个 Python 环境

仓库本身存在不可合并的 Python 版本约束：

```text
根 openpi/RLT：       requires-python >= 3.11
rlt_online_rl：       requires-python >= 3.10,<3.11
```

因此使用两个独立 uv 环境，且都放系统盘：

```text
/root/workspace/openpi-rlt-system/openpi311/.venv
    Python 3.11
    根 openpi、RLT 阶段一、Machine A

/root/workspace/openpi-rlt-system/online-rl310/.venv
    Python 3.10
    rlt_online_rl、Machine B、Replay、Learner
```

两套环境不能混用。uv 下载缓存可以放 CFS，但虚拟环境和 `site-packages` 不放 CFS。

### 2.2 系统盘与 CFS 分工

系统盘只保存可重建的运行时：

```text
/root/workspace/openpi-rlt-system/
├── openpi311/.venv/
├── online-rl310/.venv/
└── env.sh
```

CFS 保存全部长期资产：

```text
/mnt/cfs/usr/wujh/openpi-RLT/
├── repo/openpi-RLT/       # Git 代码
├── datasets/              # 原始、处理后、LeRobot 数据
├── pretrained/            # Pi0/Pi0.5 和 RLT 权重
├── checkpoints/           # 阶段一和在线 RL checkpoint
├── runs/                  # 实验运行产物
├── replay/                # replay buffer、episode journal
├── cache/                 # HF、Torch、W&B 等大缓存
├── configs/               # 实验配置副本
├── logs/                  # 安装、测试、训练和服务日志
├── reports/               # 阶段报告
└── artifacts/             # 导出模型和可视化
```

所有训练、模型、数据和日志路径必须显式指向 CFS；不能将长期资产放在 `/root/workspace`。

### 2.3 为什么第一轮选择 Pi0 + ALOHA

仓库已经包含公开 ALOHA 数据和 Pi0 数据配置：

```text
数据：lerobot/aloha_sim_transfer_cube_human
配置：pi0_aloha_sim
```

当前 AgileX 配置固定了三路相机、32 维模型动作、专用状态和专用数据字段，不能直接与 ALOHA 数据混用。因此第一轮先使用 Pi0 + ALOHA 建立可复现基线，第二轮再迁移到 Pi0.5/AgileX。

### 2.4 网络受限环境下的资产分工

2026-09-07 云端实测：PyPI、`files.pythonhosted.org`、Google Storage 和 Astral/uv 可访问；GitHub HTTPS 与 Hugging Face HTTPS 超时。由此采用分层传输策略：

```text
本地：代码修改、Git bundle、wheelhouse、源码归档、模型和数据准备
云端：创建系统盘虚拟环境、安装/导入依赖、GPU 验收、测试、训练和实验运行
```

资产传输规则：

- Git 历史使用 Git bundle；
- 普通 PyPI wheel 由云端直接下载，只有失败包才在本地制作小范围 wheelhouse；
- 默认 GitHub 依赖 `lerobot` 使用 CFS 本地 Git mirror；可选 `rlds` group 的 `dlimp` 首轮不安装；
- Hugging Face 数据由本地下载后通过 `rsync`/归档上传，或后续使用平台对象存储；
- OpenPI GCS 权重优先由云端直接下载；
- 模型、数据、checkpoint、replay 和日志使用 CFS 目录，不进入 Git；
- 不上传本地 `.venv` 或 `site-packages`，云端必须重新创建环境并验证 CUDA/JAX/PyTorch；
- 依赖安装前先判断来源，不对无法访问 GitHub/Hugging Face 的命令反复重试。

当前已经存在的 LeRobot mirror：

```text
/mnt/cfs/usr/wujh/openpi-RLT/cache/git-mirrors/lerobot.git
```

锁定 revision `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` 已存在，后续优先复用。

## 3. 算法链路

### 3.1 RLT 阶段一

```text
图像 + 状态 + 语言 + 动作样本
→ 冻结或部分冻结 Pi0 VLA
→ VLA prefix embedding
→ RLTokenEncoder
→ z_rl
→ RLTokenDecoder
→ reconstructed prefix
```

核心损失：

```text
L_RLT = MSE(reconstructed_prefix,
            stop_gradient(vla_prefix))

L_total = L_RLT + alpha * L_VLA
```

第一种实验 `alpha=0`，只训练 RLT Encoder/Decoder；第二种实验 `alpha>0`，验证联合 VLA loss 路径。RLT 的训练参数主要是轻量 Transformer，但真实数据和 VLA 权重仍然必要：数据提供观测样本，预训练 VLA 提供可压缩的 prefix 表示。

### 3.2 在线 RL 阶段二

Machine A：

```text
输入 observation
→ 冻结 Pi0 + RLT
→ 输出 z_rl + ref_chunk
```

Machine B：

```text
Actor 输入：z_rl + proprio + ref_chunk
Actor 输出：refined action chunk
Critic 输入：z_rl + proprio + action chunk
```

Actor 目标：

```text
actor_loss = bc_weight * bc_penalty
           - q_weight * actor_q
           + delta_weight * delta_penalty
```

运行顺序：

```text
VLA reference warmup
→ Replay 达到阈值
→ Learner 预训练
→ ready_for_online
→ Actor 修正 reference action
→ 环境执行 action chunk
→ Replay
→ Learner 更新
→ actor snapshot 热加载
```

## 4. 分阶段执行计划

### 阶段 0：云端基础环境

1. 通过 SSH 别名 `openpi-rlt` 登录开发机；
2. 只读检查系统盘、CFS、GPU、CPU、内存、Python 和 uv；
3. 安装 `git`、`tmux` 等基础工具；
4. 在系统盘创建 `/root/workspace/openpi-rlt-system/env.sh`，将代码、数据、权重和缓存路径指向正确位置；
5. 用 Git 验证 CFS 现有仓库和待执行提交一致；只有仓库缺失时才 clone 到 `repo/openpi-RLT`；
6. 确认代码在 CFS、两个虚拟环境在系统盘，目录没有混淆。

验收：CFS repo 可写、系统盘空间充足、实际 GPU 可见、代码提交可确认、环境路径符合系统盘/CFS 分工。

### 阶段 1：两个 uv 环境和依赖资产

1. 用 Python 3.11 创建根环境；
2. 用 Python 3.10 创建在线 RL 环境；
3. 将 Hugging Face、W&B、uv 下载缓存指向 CFS；
4. 先检查 PyPI、GitHub、Hugging Face 的可达性；
5. 根项目优先使用 PyPI + CFS Git mirror，受限依赖使用本地 wheelhouse/源码归档；
6. 不复制本地 `.venv`，不把模型、数据和 wheelhouse 放入 Git；
7. 安装完成后严格验证两个解释器、包导入和 GPU 后端。

验收：

```text
Python 3.11 环境能导入 openpi/jax/flax/torch；
Python 3.10 环境能导入 rlt_online_rl/jax/flax/optax；
两套环境的 python 路径均在 /root/workspace。
```

### 阶段 2：已有测试与 GPU 验证

按顺序运行：

```text
GPU/JAX/PyTorch 设备测试
→ 根项目轻量测试
→ rlt_online_rl 全部单测
→ 完整根项目测试
```

长任务使用 `tmux`，日志写入 CFS。先记录失败原因，不把显存、依赖、测试逻辑错误混为一谈。

### 阶段 3：fake RLT smoke training

运行 `debug_rlt` 和 `debug_rlt_joint`，验证：

- dummy Pi0 构建；
- prefix embedding；
- RLT Encoder/Decoder；
- loss 与反向传播；
- checkpoint 保存、恢复；
- `alpha=0` 和 `alpha>0` 两条路径。

fake 数据只证明工程链路，不证明表示学习效果。

### 阶段 4：公开 ALOHA 数据和 Pi0 权重

使用仓库已有公开路线：

```text
数据：lerobot/aloha_sim_transfer_cube_human
权重：gs://openpi-assets/checkpoints/pi0_base/params
```

逐项检查：

- dataset sample 的 keys、dtype、shape；
- 图像相机字段和分辨率；
- ALOHA state 与 action 维度；
- `AlohaInputs/Outputs` 的坐标和夹爪转换；
- action padding 到 Pi0 内部维度；
- norm stats 和 assets 路径；
- 权重加载和缓存落盘位置。

### 阶段 5：Pi0 + ALOHA 的 RLT 配置

在不破坏现有 `pi0_aloha_sim` 的前提下增加对应的 RLT 配置：

```text
rlt_pi0_aloha
rlt_pi0_aloha_joint
```

先运行冻结 VLA 的 `alpha=0`，再运行联合损失版本。每次使用明确的实验名和 CFS checkpoint 目录。

训练阶梯：

```text
20 steps → 100 steps → 1,000 steps → 5,000 steps
```

每一级检查 loss、显存、checkpoint、恢复和日志。

### 阶段 6：Machine A 与部署 contract

使用 `scripts/serve_rlt_policy.py` 加载 RLT checkpoint，并在联调前解决任务动作 contract：

```text
输入 observation
→ z_rl
→ proprio
→ ref_chunk
```

当前 `AlohaOutputs` 返回 14 维双臂动作，而服务端固定输出 7 维动作、50-step chunk，AgileX online RL 配置消费 7 维动作、10-step chunk。Pi0 + ALOHA 可先作为 RLT 阶段一训练基线，但进入在线 RL 前必须通过参数化服务输出或独立 ALOHA/ManiSkill adapter 明确动作维度、关节顺序、单位和语义，不能只依赖数组裁剪。

当前服务还硬编码监听 `0.0.0.0` 且没有 `--host`。云端启动前应增加本机监听选项，或确认平台网络层阻止公网访问。之后再检查模型加载、单请求、重复请求、batch、超时、断连和重启恢复。

### 阶段 7：fake Machine A + fake 环境

先不引入 ManiSkill，使用 fake payload 和确定性 fake 环境验证：

```text
observation
→ z_rl/ref_chunk
→ Actor
→ action chunk
→ reward/done
→ Replay
→ Learner
→ snapshot
→ Actor 热加载
```

验收：Replay 增长、Learner `global_step` 增长、Actor version 增长、snapshot 生成、服务重启可恢复。

**2026-09-08 云端完成记录：** 使用仅监听 `127.0.0.1` 的 fake Machine A、确定性 7 维环境和小预算配置完成三进程 Machine B 闭环。两个 warmup episode 写入 12 条 transition，Learner 完成 2 次更新并产生 Actor `v1/v2` snapshot 与 `step_2` checkpoint；第三个 post-update episode 确认 Actor 已热加载 version 2。停止并重启四个服务后，Replay 恢复到 18 条、Actor version 保持 2、Learner 从 `global_step=2` 的 checkpoint 恢复且保持冻结。该结果只证明软件/RPC/持久化链路，不代表真实任务的算法效果。

### 阶段 8：Machine B

在同一台 B300 上以多进程运行：

```text
Machine A
Actor service
Learner service
Replay service
rollout/environment
```

首轮使用小规模 warmup 和 update，不直接采用生产配置的 600 条 warmup、20,000 次预训练更新等大预算。确认进程间 RPC、Replay journal、learner checkpoint 和 snapshot 后再增加预算。

### 阶段 9：ManiSkill 适配

新增独立仿真适配层，不破坏 ROS/AgileX 路径。最小接口：

```python
reset() -> observation
step(action) -> next_observation, reward, done, info
execute_chunk(action_chunk) -> execution result
```

第一版选择稳定的单臂任务和 7 维动作接口，负责 observation/proprio、action chunk、reward/done、reset 和 success 转换。先单独验证 adapter，再接 fake Machine A，最后接真实 Machine A。

### 阶段 10：仿真在线 RL 与对照实验

至少运行：

```text
reference-only baseline
Actor refinement online RL
```

保存：

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

生成阶段报告，明确数据、模型、配置、硬件、失败案例和不能外推到真实机器人的结论。

### 阶段 11：镜像和分布式扩展

只有在 uv 环境、训练、服务和仿真闭环稳定后才制作 Docker 镜像。镜像包含 Python、系统工具、项目依赖和 CUDA 用户态运行时，不包含数据、权重、checkpoint、replay 和大缓存。多 GPU/多机时使用相同镜像、各机本地环境和共享 CFS 资产。

## 5. 目录、日志和安全要求

- 代码只放 `.../repo/openpi-RLT`；
- 训练 checkpoint 只放 `.../checkpoints`；
- 模型和数据只放 `.../pretrained`、`.../datasets`；
- 长日志只放 `.../logs`；
- 不把大文件放 `/root/workspace`；
- 不操作 `/mnt/cfs/usr` 下其他用户目录；
- 删除 checkpoint、清理 replay、使用 `--overwrite`、停止训练前必须再次确认；
- token、私钥和账号凭证不进入仓库；
- 长命令使用 `tmux` 或 `nohup`，避免 SSH 断开导致任务退出。

## 6. 当前执行顺序

截至 2026-09-07，已经完成：

```text
本机双 uv 环境和轻量软件验证
SSH 别名 openpi-rlt
云端 GPU、CPU、内存、系统盘和 CFS 只读勘测
CFS 个人目录架构
云端仓库已存在，main 指向提交 189a28d
云端 uv 0.10.12 可用
路线选择：Pi0 + ALOHA
仿真路线：fake 环境 → ManiSkill
```

## 6. 当前执行状态（2026-09-08）

下表是本计划的唯一当前状态摘要；以它替代早期“尚未完成”的环境安装描述。

| 阶段 | 状态 | 已完成或下一关卡 |
|---|---|---|
| 阶段 0：云端基础环境 | **完成** | SSH、git、tmux、CFS 目录、系统盘/CFS 分工、GPU 与磁盘检查完成。 |
| 阶段 1：双环境与依赖资产 | **完成** | 根 Python 3.11 与 Online RL Python 3.10 分离；Git mirror、PyTorch wheelhouse、JAX 私有 CUDA runtime、Online RL 离线归档均已准备。 |
| 阶段 2：已有测试与 GPU 验证 | **部分完成** | 根环境的 JAX/PyTorch GPU 实算通过；Online RL `47 passed`；完整根项目测试尚未运行，不能标记完成。 |
| 阶段 3：fake RLT smoke | **完成** | `debug_rlt`、checkpoint 恢复、`debug_rlt_joint` 均通过并写入 CFS。 |
| 阶段 4：公开 ALOHA 数据和 Pi0 权重 | **完成** | ALOHA v3 数据与 `pi0_base` 参数已校验并导入 CFS；已完成 14D schema、50Hz、50 episodes 和 Trossen norm stats 审计。 |
| 阶段 5：Pi0 + ALOHA RLT 配置 | **完成 smoke** | 新增 `rlt_pi0_aloha`/`rlt_pi0_aloha_joint`；Pi0 + ALOHA 20-step RLT smoke 及从 step 19 恢复到 step 21 已通过。正式 100/1,000/5,000 steps 未启动。 |
| 阶段 6：Machine A 与部署 contract | **完成 smoke** | 真实 Pi0/RLT Machine A 在 `127.0.0.1:8000` 运行；输出 `z_rl[2048]`、完整 14D 输入下的 `proprio[7]` 与 `ref_chunk[50,7]`，healthz 和 batch contract 已验证。 |
| 阶段 7：fake Machine A + fake 环境 | **完成** | 云端 bounded smoke 已验证 fake Machine A、Actor、Replay、Learner、热加载、checkpoint/replay 恢复与持久化工件。 |
| 阶段 8：Machine B 多进程 | **完成仿真 smoke** | 真实 Machine A + Actor + Replay + Learner + Gym-ALOHA 单臂 7D adapter 已完成 2 episode 闭环，Learner 到 `global_step=4`，Actor version 到 `2`。 |
| 阶段 9：仿真 adapter | **完成 Gym-ALOHA 版本** | 已完成可无窗口运行的 Gym-ALOHA 单臂 7D adapter；本轮未引入 ManiSkill，避免把两个仿真后端混为同一验收。 |
| 阶段 10：仿真 Online RL 对照实验 | **完成 bounded smoke** | 已跑通真实 Machine A reference、Actor refinement、Replay、Learner、snapshot、checkpoint 和 episode metrics；不是任务成功率基线或大规模算法结论。 |
| 阶段 11：镜像和分布式扩展 | **完成镜像导出** | `openpi-rlt:sim-ready` 已构建并通过容器内 JAX GPU、Gym-ALOHA import/reset 验证；镜像 tar.gz 导出资产正在完成校验。 |

### 6.1 已完成工作与证据

```text
[完成] 云端根 RLT 环境：Python 3.11.15、JAX 0.5.3、torch 2.10.0+cu128
[完成] sm_120 上 JAX 与 PyTorch 的实际 GPU 矩阵乘法
[完成] RLT fake-data 训练、checkpoint 保存/恢复、联合损失路径
[完成] 云端 Online RL 环境：Python 3.10.20、JAX/Flax/Optax/OpenCV
[完成] Online RL JAX GPU 矩阵乘法和 47 项测试
[完成] CFS 日志、checkpoint、离线 wheelhouse/归档与 Git bundle 同步流程
[完成] 根项目 uv.lock 与 Online RL uv.lock 可校验
[完成] 确定性 fake 环境、fake Machine A healthz/batch contract 与 47 项 Online RL 测试
[完成] 云端 fake 闭环：3 个 episode、18 条 replay transition、Learner global_step=2、Actor version=2
[完成] fake 闭环持久化：raw episode、Replay journal、Actor snapshot history、Learner checkpoint、metrics/logs
[完成] fake 服务停止/重启：Replay=18、Actor version=2、Learner global_step=2 恢复
[完成] 公开 ALOHA v3 数据：50 episodes、20,000 frames、14D state/action、50Hz；本地与云端 SHA256 一致
[完成] Pi0 base 参数：33 个文件、约 11.19 GiB；本地与云端文件大小一致
[完成] `rlt_pi0_aloha` 20-step smoke、checkpoint step 19，以及恢复到 step 21
[完成] 真实 Pi0/RLT Machine A：`z_rl=(2048,)`、`proprio=(7,)`、`ref_chunk=(50,7)`，监听 `127.0.0.1:8000`
[完成] Gym-ALOHA 单臂 7D adapter：显式 left/right 映射，另一只手保持当前状态，不做隐式数组截断
[完成] 真实 Machine A + Gym-ALOHA + Machine B：2 episodes、62 transitions、Learner `global_step=4`、Actor version=2
[完成] Docker `openpi-rlt:sim-ready` 构建，容器内 JAX GPU 与 Gym-ALOHA smoke
```

核心证据目录：

```text
RLT 日志：/mnt/cfs/usr/wujh/openpi-RLT/logs/training/
Online RL 最终测试日志：/mnt/cfs/usr/wujh/openpi-RLT/logs/tests/online-rl310-fake-smoke-final-20260908.log
Fake 闭环运行目录：/mnt/cfs/usr/wujh/openpi-RLT/runs/online_rl/fake-machine-a-smoke-20260908-final/
RLT checkpoint：/mnt/cfs/usr/wujh/openpi-RLT/checkpoints/rlt_stage1/
```

### 6.2 当前结论与可选后续

本轮已完成“Pi0 + ALOHA 数据/RLT + 真实 Machine A + Gym-ALOHA 单臂 7D Online RL + Docker 镜像”的软件闭环。当前不把它表述为 ManiSkill 结果，也不把 bounded smoke 表述为真实机器人成功率。

正式训练和扩展可以从以下顺序开始：

```text
1. 视资源情况继续 rlt_pi0_aloha：100 → 1,000 → 5,000 steps
2. 运行 reference-only 与 Actor refinement 的固定种子对照集
3. 若必须使用 ManiSkill，再新增独立 ManiSkill adapter 和对应验收
4. 将已导出的 sim-ready 镜像上传到分布式训练节点，使用各节点 CFS 挂载数据/权重/checkpoint
5. 单卡基线稳定后，再规划 FSDP、多 GPU、Pi0.5/AgileX 和真实机器人
```

在线 RL 不使用 PyTorch；它通过 JAX CUDA 使用 GPU。PyTorch 的 `sm_120` 兼容问题仅属于根 OpenPI/RLT Python 3.11 环境。ROS 2 / `rclpy` 是真实机器人阶段的系统级依赖，不属于当前 uv venv。

完整的可复制命令、路径和成功判据见
[CLOUD_DEPLOYMENT_GUIDE.zh-CN.md](CLOUD_DEPLOYMENT_GUIDE.zh-CN.md)。
