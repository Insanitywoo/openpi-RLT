# RLT 在线强化学习运行时

本文档介绍 `rlt_online_rl` 运行时：它是在 openpi/RLT 模型训练并部署为服务之后使用的轻量级在线强化学习系统。仓库根目录的 [README](../README.md) 介绍项目概览、演示视频、RLT 与 openpi 的关系、贡献者和引用模板。

`rlt_online_rl` 实现面向机器人的在线学习闭环：

- **Machine A** 特征/参考服务：提供 `z_rl` 和 VLA 参考动作块；
- **Machine B**：Actor 服务、Learner 服务及 Replay Manager；
- 机器人 rollout 驱动：连接 ROS 观测、Machine A、Actor、回放、人工信号、复位与评估。

目前公开的任务配置是 `configs/tasks/agilex_ethernet`，展示真实机器人上的以太网插接。

## 范围

此软件包负责在线 RL 运行时：

- B1 `actor_service`；
- B2 `learner_service`；
- B3 `replay_manager`；
- B4 `EnvDriver`；
- ROS rollout 适配器；
- 回放日志（journal）、原始回合持久化、Actor 快照、日志和指标；
- warmup、在线 rollout、人工接管、关键阶段切换，以及仅评估执行。

它**不**训练基础 VLA 或 RL-token 模块；这些内容位于仓库根目录的 openpi 栈中，主要入口是 `src/openpi`、`scripts/train_rlt.py` 和 `scripts/serve_rlt_policy.py`。

## 运行时架构

**Machine A** 运行冻结的 openpi/RLT 策略服务器。对每个观测，它返回：

- `z_rl`：紧凑的 RL-token 特征；
- `ref_chunk`：VLA 生成的参考动作块。

**Machine B** 运行：

- `actor_service`：提供当前轻量级 Actor，以低延迟方式精修动作；
- `learner_service`：从回放中采样、训练 Actor/Critic，并发布 Actor 快照；
- `replay_manager`：维护回放缓冲区和追加式 journal。

机器人 rollout 会：

- 读取 ROS 观测；
- 查询 Machine A；
- 执行 VLA 参考动作块或 Actor 精修后的动作块；
- 记录原始逐步轨迹；
- 在回合结束后构建 replay transition；
- 将 transition 发送给 Replay Manager。

## 动作块执行路径

每一个动作块边界上，执行如下步骤：

1. rollout 适配器读取当前机器人观测；
2. 将观测发送到 Machine A；
3. Machine A 返回 `z_rl` 和 `ref_chunk`；
4. rollout 从本地观测状态中提取 `proprio`；
5. 在 warmup 或 `full_task` 的非关键前缀阶段，直接执行 `ref_chunk`；
6. 在在线关键阶段控制中，Machine B 的 Actor 接收 `z_rl / proprio / ref_chunk` 并返回精修后的动作块；
7. 机器人连续执行选中动作块共 `chunk_exec_horizon` 个控制周期；
8. 本回合首先保存实际执行的原始 step；
9. 回合结束后构建回放窗口，并回填缺失的 Machine A 锚点特征；
10. Learner 从回放采样并发布 Actor 快照；
11. Actor 服务热加载最新快照。

## 核心模式

### Warmup

Warmup 使用冻结 VLA 参考策略收集回放数据，此时不允许 Actor 控制机器人。回放达到 `warmup_min_size` 后，Learner 才开始训练。

若设置了 `warmup_post_collect_updates`，在允许在线 rollout 前，Learner 会先完成指定次数的 warmup 更新。否则，所需 warmup 更新预算由 `warmup_ready_adds_total * grad_updates_per_cycle` 推导。

### Warmup 等待在线就绪

收集到足够 warmup 数据后，rollout 会同时等待：

- Learner 状态中的 `ready_for_online == true`；
- Actor 版本达到或超过 rollout 所要求的阈值。

只会在**回合之间**切换到在线控制，绝不会在当前回合的中途切换。

### Online

在线回合的关键阶段可以使用 Actor。训练时 Actor 是随机还是确定性的，取决于 `runtime.env_driver.actor_deterministic`；仅评估 rollout 强制使用 Actor 均值，因此是确定性的。

### 关键阶段与完整任务

`critical_phase` 表示回合直接从高精度关键动作段开始。`full_task` 则会先使用基础策略执行关键段之前的流程，接收到人工关键阶段信号后再切换至关键控制。`full_task` 的非关键前缀不会写入回放。

## 当前以太网任务默认值

当前以太网任务配置使用：

- `action_dim: 7`
- `chunk_len: 10`
- `z_dim: 2048`
- `proprio_dim: 7`
- `action_representation: delta_chunk`
- `reference_dropout_prob: 0.5`
- `warmup_min_size: 600`
- `warmup_post_collect_updates: 20000`
- `grad_updates_per_cycle: 5`
- `step_trace_stride: 0`
- `control_frequency_hz: 20.0`

注意：`step_trace_stride: 0` 会禁用稠密 stride replay，仅保留动作块边界处的回放。这是当前以太网配置有意采用的运行时设定。

## 回放语义

每条 replay transition 包含：

- `z_rl`、`proprio`；
- `ref_chunk`：该 transition 对应观测下 Machine A/VLA 的参考动作；
- `action_chunk`：机器人实际执行的动作；
- `rewards`、`done`；
- `next_z_rl`、`next_proprio`、`next_ref_chunk`；
- `source`：动作块级控制来源；
- `source_chunk`：逐动作步的控制来源；
- `collection_phase`：warmup 或 online；
- `episode_id`、`step_id`、`success`、`intervention_flag`。

`TransitionSource` 只是一个控制来源标签：

- `BASE`：执行冻结 VLA 参考动作；
- `RL`：执行 Actor 精修动作；
- `HUMAN`：人工控制；
- `MIXED`：同一窗口同时包含人工与策略控制步骤。

Trainer 使用 `source_chunk` 逐步选择 BC 目标：

- `HUMAN / MIXED` 步使用实际执行的 `action_chunk` 作为模仿目标；
- `BASE / RL` 步使用 VLA 的 `ref_chunk` 作为模仿目标。

这与“用人工动作替换 `ref_chunk`”不同。部署时 Actor 仍会看到 VLA 参考动作；因此人工数据教会的是：**如何把 VLA 参考动作修正成真实执行的人工校正动作**。

## Learner 目标

Learner 使用双 Critic、固定标准差的高斯动作块 Actor、目标网络及参考动作 dropout。当前 Actor 损失为：

```text
actor_loss = bc_weight * bc_penalty - q_weight * actor_q + delta_weight * delta_penalty
```

Warmup 和在线阶段可使用不同的 BC/Q 权重：

- `warmup_bc_weight`、`warmup_q_weight`；
- `online_bc_weight`、`online_q_weight`。

`delta_penalty` 会先将归一化训练动作转换回可执行的绝对动作块，然后比较前六个机械臂关节相邻 step 的增量。

## 回放窗口

回放在回合结束后由原始回合轨迹构建。

`step_trace_stride: 0` 时：

- 使用动作块边界 replay window；
- 当人工控制重新交回策略控制时，增加策略重启锚点；
- 可能增加一个与终止点对齐的最终窗口；
- 只回填这些窗口所需的锚点。

`step_trace_stride > 0` 时：

- 从原始逐 step 轨迹按照设定 stride 构建稠密 replay window；
- 对缺失锚点使用批量 Machine A 特征回填；
- 取 `2` 时与 RLT 论文所用的稠密 replay 思路一致。

Replay journal 是追加式 pickle 流。启动时 Replay Manager 会从中恢复，并从已恢复最大 `episode_id + 1` 继续编号。

## 人工控制与手动信号

ROS 适配器支持以下人工服务：

- 请求开始下一个回合；
- 记录成功、失败或结束；
- 进入或切换关键阶段；
- 选择下一关键阶段使用 Actor 或基础策略；
- 切换遥操作接管。

人工接管期间，回放会记录每个控制 tick 最新采样到的人工动作；不会将原始遥操作事件流直接写入回放。

## 安装

在线 RL 运行时请使用一个单独的 Python 3.10 环境：

```bash
cd openpi-RLT/rlt_online_rl
conda create -y -n rlt_online_rl310 python=3.10 pip
conda activate rlt_online_rl310
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ../packages/openpi-client
python -m pip install -e .
```

机器人 rollout 和键盘客户端也要求在各自运行的 shell 中 source ROS：

```bash
source /opt/ros/humble/setup.bash
```

可选的 W&B 辅助监控支持：

```bash
python -m pip install -e '.[monitor]'
```

## 启动训练

启动 Machine B 服务：

```bash
cd openpi-RLT/rlt_online_rl
conda activate rlt_online_rl310
python launch/launch_machine_b.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml
```

RLT checkpoint 准备好后，从仓库根目录启动 Machine A：

```bash
cd openpi-RLT
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir <checkpoint-dir> \
  --port 8000 \
  --shared-prefix-inference
```

`--shared-prefix-inference` 是 Machine A 服务器的**仅推理阶段**延迟优化。Machine A 同时需要 `z_rl` 和冻结 VLA 的 `ref_chunk`：旧路径会先为 `z_rl` 计算一次 VLA prefix，并在动作采样内部为了构建 KV cache 再计算一次。使用该参数时，服务器只计算一次 prefix，并复用同一份 prefix 输出/KV cache 来生成两种输出。它不会改变训练、checkpoint、模型权重、归一化或在线 RL 运行时 payload。若要严格复现旧推理路径，可不使用此参数。

没有真实 VLA 服务时，可使用下列命令进行本地集成测试：

```bash
cd openpi-RLT/rlt_online_rl
python launch/fake_machine_a.py
```

纯软件闭环还可使用 `rlt_online_rl.fake_env:DeterministicChunkEnv`。该环境不导入 ROS、不连接机器人；它消费 Actor 输出的动作块并生成确定性的 observation、reward、done 与 EnvDriver step trace。启动时必须将 `local_debug_mode` 设为 `false`，这样 EnvDriver 才会经 WebSocket 调用 fake Machine A，而不是使用内置 `DummyFeatureProvider`。

启动机器人 rollout：

```bash
cd openpi-RLT/rlt_online_rl
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python launch/launch_robot_rollout.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000
```

启动训练键盘客户端：

```bash
python keyboard_toggle_teleop_record_reward_isolation.py
```

## 启动评估

评估不启动 Learner 或 Replay；只运行 Actor 推理和机器人 rollout。

启动 Actor 服务：

```bash
cd openpi-RLT/rlt_online_rl
conda activate rlt_online_rl310
python scripts/run_online_rl.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --system.role actor_service \
  --system.actor_service.snapshot_path <actor_snapshot.pkl>
```

启动评估 rollout：

```bash
python launch/launch_actor_eval.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000
```

`launch_actor_eval.py` 会等待 Actor 服务就绪，再启动 `pika_sync_ros.py --eval_actor_only`。仅评估 rollout 强制使用确定性的 Actor 均值。

启动评估键盘客户端：

```bash
python keyboard_actor_eval.py
```

## 常用工具

检查回放：

```bash
python scripts/tools/inspect_replay_journal.py \
  runs/agilex_ethernet/replay/replay_journal.pkl
```

绘制 Learner 指标：

```bash
python scripts/tools/plot_learner_metrics.py \
  --run_dir runs/agilex_ethernet
```

离线训练与分析工具见 [scripts/offline/README.md](scripts/offline/README.md)。

真实机器人回放导出与播放工具见 [scripts/replay_real_robot/README.md](scripts/replay_real_robot/README.md)。

## 建议的首次运行顺序

训练：

1. 启动 Machine B 服务；
2. 启动 Machine A；
3. 启动机器人 rollout；
4. 确认机器人已复位到起始姿态；
5. 启动训练键盘客户端；
6. 按 `o` 开始回合；
7. 若使用 `full_task`，在关键阶段边界按 `c`；
8. 成功按 `s`，失败按 `f`。

评估：

1. 启动 Actor 服务；
2. 启动评估 rollout；
3. 确认机器人已复位到起始姿态；
4. 启动评估键盘客户端；
5. 按 `a` 或 `b`，为下一次关键阶段选择 Actor 或基础策略；
6. 按 `o` 开始回合；
7. 若使用 `full_task`，在关键阶段边界按 `c`；
8. 在回合应结束时按 `s`。

## 常见误解

- `full_task` 的前缀在关键阶段开始前不会写入回放；
- `full_task` 的前缀不使用 Actor，而执行 Machine A 的参考动作；
- 训练中 `s` 表示成功并结束回合；
- 评估中 `s` 仅结束/复位回合，**不是**训练奖励；
- 仅评估 rollout 会忽略训练 rollout 的随机性设置，使用 Actor 均值；
- 评估中的 `a / b` 决定的是**下一回合**关键阶段的策略，并非当前回合中途立即切换；
- `critical_phase` 通常不需要按 `c`，因为它开始时已经在关键段内；
- warmup 就绪绝不会在当前回合中途切换到在线控制。

## 目录结构

```text
rlt_online_rl/
|-- configs/                    # 基础配置与任务运行时配置
|-- launch/                     # Machine B、rollout、评估与假 Machine A 启动器
|-- scripts/offline/            # 离线回放训练和分析
|-- scripts/replay_real_robot/  # 导出和播放参考/Actor 关节动作块
|-- scripts/tools/              # 轻量级检查与绘图工具
|-- src/rlt_online_rl/          # 核心运行时软件包
|-- train_deploy_alignment/     # ROS 适配器和人工信号桥接
`-- tests/                      # 运行时单元测试
```
