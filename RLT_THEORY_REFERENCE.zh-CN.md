# openpi-RLT：RLT 理论与端到端链路参考

> 本文档用于记录 `openpi-RLT` 项目的核心理论、系统链路和算法逻辑，作为后续阅读代码、配置实验和排查运行问题时的参考。
>
> 相关实现主要位于：
>
> - `src/openpi/models/rl_token.py`
> - `scripts/train_rlt.py`
> - `scripts/serve_rlt_policy.py`
> - `rlt_online_rl/src/rlt_online_rl/`
> - `rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml`

## 1. 项目要解决什么问题

`openpi-RLT` 在 openpi/pi0.5 的视觉-语言-动作（VLA）基础上，引入 RL Token（RLT）和轻量级在线 Actor-Critic，使真实机器人能够在执行 VLA 基础策略的同时，根据真实环境反馈学习动作修正。

它不是让强化学习从零开始学习机器人控制，而是让 RL 学习：

> 如何根据当前任务状态和 VLA 的参考动作，对 VLA 动作块进行局部、安全、平滑且高回报的修正。

这类设计尤其适合以太网插接、精细对准、接触操作等任务：VLA 负责提供通用的视觉-语言先验，在线 RL 负责处理真实机器人、工件位置和接触状态带来的局部误差。

## 2. 总体架构

```text
阶段 1：RLT 模型训练

示范数据
  │
  ▼
pi0.5 / VLA 提取视觉-语言前缀特征
  │
  ▼
RL-token Encoder：压缩 VLA 前缀特征
  │
  ▼
RL-token Decoder：重建 VLA 前缀特征
  │
  ▼
得到可供在线 RL 使用的紧凑任务特征 z_rl


阶段 2：真实机器人在线 RL

机器人观测 ──► Machine A：冻结的 VLA + RLT 策略服务
                         │
                         ├──► z_rl：RLT 紧凑状态特征
                         └──► ref_chunk：VLA 参考动作块
                                      │
                         ┌────────────┘
                         ▼
                 Machine B：轻量 Actor 服务
                 输入 z_rl / proprio / ref_chunk
                         │
                         ▼
                 refined_chunk：精修后的动作块
                         │
                         ▼
                    真实机器人执行
                         │
                         ▼
             回放、Critic 学习、Actor 更新、快照热加载
```

## 3. 两阶段训练与部署流程

### 3.1 阶段 1：训练 RL-token

给定图像、语言指令和机器人观测，VLA 会产生一段高维 prefix embedding：

```text
prefix_embs: [batch, sequence_length, input_dim]
```

当前默认的 VLA 隐状态输入维度是 `2048`。直接把完整 prefix 送给在线 RL 会带来较高的计算、通信和存储成本，因此 RLT 使用少量可学习 token 对其进行压缩。

#### RL-token Encoder

Encoder 使用可学习 query token 对 VLA prefix embeddings 做交叉注意力：

```text
prefix_embs
   │
   ▼
Cross-Attention Encoder
   │
   ▼
rl_tokens: [batch, num_rl_tokens, embed_dim]
```

默认配置包括：

```text
num_rl_tokens = 1
embed_dim = 512
input_dim = 2048
```

直观地说，Encoder 从 VLA 的长序列内部表示中提取一个紧凑的任务状态摘要。该摘要在服务端展平后作为 `z_rl` 提供给在线 RL。

#### RL-token Decoder

仅压缩而不加约束可能导致信息丢失，因此 RLT 使用 Decoder 从 RL tokens 重建原始 prefix embedding：

```text
rl_tokens
   │
   ▼
Cross-Attention Decoder
   │
   ▼
reconstructed_prefix_embs
```

重建损失为：

```text
L_RLT = || Decoder(Encoder(h_VLA)) - stop_gradient(h_VLA) ||²
```

代码中对 VLA 特征使用 `stop_gradient`，因此 RL-token 重建损失不会通过目标特征反向改变 VLA 表示。

#### 是否微调 VLA

阶段 1 支持两种模式：

- **冻结 VLA**：只训练 RL-token Encoder/Decoder。
- **联合训练**：在 RLT 重建损失之外加入监督式 VLA 损失：

```text
L_total = L_RLT + alpha * L_VLA
```

其中 `alpha` 控制 VLA 监督损失的权重。

### 3.2 阶段 2：冻结 VLA，在线训练轻量 Actor-Critic

阶段 1 获得的 checkpoint 被部署到 Machine A。Machine A 对每次机器人观测返回：

```python
{
    "z_rl": z_rl,
    "ref_chunk": ref_chunk,
}
```

- `z_rl`：从 VLA prefix 压缩得到的任务状态特征；
- `ref_chunk`：冻结 VLA 生成的参考动作块。

在线 RL 不再重新处理图像和语言，而是在这些输出之上学习动作修正。

## 4. 一次动作块控制周期

当前以太网任务默认配置：

```yaml
action_dim: 7
chunk_len: 10
chunk_exec_horizon: 10
control_frequency_hz: 20.0
```

一次动作块大致是 `[10, 7]`，机器人执行 10 个控制 tick 后重新观测和规划。

控制流程：

1. ROS 读取当前机器人观测；
2. rollout 适配器将观测发送给 Machine A；
3. Machine A 返回 `z_rl` 和 `ref_chunk`；
4. 本地从机器人状态提取 `proprio`；
5. warmup 或非关键阶段直接执行 `ref_chunk`；
6. 在线关键阶段将 `z_rl / proprio / ref_chunk` 发送给 Actor；
7. Actor 返回精修动作块 `refined_chunk`；
8. 机器人执行选中的动作块；
9. 系统记录实际执行的原始 step；
10. 回合结束时从原始轨迹构建 replay transition；
11. Learner 采样 replay 并更新 Actor/Critic；
12. Learner 发布新 Actor 快照，Actor 服务热加载。

## 5. 在线 Actor 的输入与输出

Actor 接收：

```text
z_rl       ：VLA/RLT 的视觉-语言任务特征
proprio    ：机器人本体状态，例如关节位置
ref_chunk  ：VLA 建议执行的动作块
```

输出一个动作块分布：

```text
pi(a | z_rl, proprio, ref_chunk)
  = Normal(mu_theta, fixed_std² I)
```

当前 Actor 使用固定标准差的高斯分布：

- 训练 rollout 可以随机采样，以保留探索；
- 评估 rollout 强制使用均值 `mu_theta`，保证确定性。

Actor 网络会分别投影三种输入，再拼接后经过 MLP：

```text
z_rl      → 256 维
proprio   → 64 维
ref_chunk → 256 维
```

最后输出 `chunk_len * action_dim` 个数值，并 reshape 成动作块。

## 6. 动作表示与归一化

当前以太网任务使用：

```yaml
action_representation: delta_chunk
```

这表示训练时动作主要使用相对于当前机器人状态的增量，而不是直接使用绝对关节目标。

对前六个机械臂关节，可近似表示为：

```text
chunk_delta[k] = chunk_abs[k] - state0
chunk_abs[k]   = chunk_delta[k] + state0
```

同时动作会使用 quantile 统计量进行归一化，部署执行前再反归一化并还原为绝对动作块。

这样做可以：

- 降低不同初始姿态对训练的影响；
- 统一动作数值尺度；
- 让 Actor 更关注“相对当前状态修正多少”。

## 7. Warmup 与在线切换

在线 RL 开始时，Actor 不立即控制机器人。Warmup 阶段只使用冻结 VLA：

```text
执行 ref_chunk
      │
      ▼
收集 replay
      │
      ▼
达到 warmup_min_size
      │
      ▼
Learner 开始 warmup 更新
      │
      ▼
Actor 达到要求版本
      │
      ▼
后续回合允许在线控制
```

当前以太网默认值：

```yaml
warmup_min_size: 600
warmup_post_collect_updates: 20000
grad_updates_per_cycle: 5
```

系统等待以下两个条件同时满足：

- Learner 报告 `ready_for_online == true`；
- Actor 版本达到 rollout 要求的阈值。

切换只发生在回合之间，不会在一个正在执行的回合中途切换控制权。

## 8. Chunk-level Twin Critic

### 8.1 Critic 输入

Critic 估计动作块级别的价值：

```text
Q(z_rl, proprio, action_chunk)
```

项目使用两个独立的 Q 网络：

```text
Q1(s, a), Q2(s, a)
```

计算目标时使用：

```text
min(Q1, Q2)
```

这样可以缓解单个 Critic 的 Q 值过估计。

### 8.2 Chunk TD 目标

一个 transition 对应一个动作块，块内包含多个 reward。首先计算块内折扣回报：

```text
R_chunk = r_0 + gamma*r_1 + gamma²*r_2 + ...
```

目标 Q 值为：

```text
y = R_chunk + (1 - done) * gamma^H * min(Q1_target(s', a'), Q2_target(s', a'))
```

其中：

- `H` 是动作块长度；
- `done` 表示回合是否结束；
- `a'` 是目标 Actor 在下一状态采样的动作块；
- `Q_target` 是目标 Critic。

Critic 损失为：

```text
L_critic = mean((Q1 - y)²) + mean((Q2 - y)²)
```

目标 Actor 和目标 Critic 使用 soft update：

```text
target ← (1 - tau) * target + tau * online
```

## 9. Actor 损失：BC + Q + Delta

项目当前使用的 Actor 损失是：

```text
actor_loss = bc_weight * bc_penalty
           - q_weight * actor_q
           + delta_weight * delta_penalty
```

### 9.1 BC 惩罚

基础行为克隆项为：

```text
bc_penalty = mean((actor_action - bc_target)²)
```

BC 目标根据控制来源逐 step 决定：

| 来源 | BC 目标 |
|---|---|
| `BASE` | VLA 的 `ref_chunk` |
| `RL` | VLA 的 `ref_chunk` |
| `HUMAN` | 实际执行的人工 `action_chunk` |
| `MIXED` | 每一步根据 `source_chunk` 选择对应目标 |

因此对正常策略数据，Actor 被约束在 VLA 参考动作附近；对人工接管数据，Actor 学习人工对 VLA 动作做出的修正。

关键点是：Actor 部署时仍然看到 VLA 的 `ref_chunk`，所以它学习的不是完全替代 VLA，而是学习一个条件化的动作编辑器。

### 9.2 Q 优化项

```text
-q_weight * actor_q
```

最小化 Actor loss 等价于推动 Actor 输出更高 Q 值的动作，从而提高预期回报。

### 9.3 Delta 平滑惩罚

系统先将归一化动作还原为可执行的绝对动作，然后比较动作块内部相邻 step 的变化：

```text
delta_penalty = mean(
    ((pred_abs[k+1] - pred_abs[k])
     - (target_abs[k+1] - target_abs[k]))²
)
```

当前主要比较前六个机械臂关节，用于抑制动作块内部的突然跳变，使动作更加平滑。

## 10. Reference Dropout

当前默认：

```yaml
reference_dropout_prob: 0.5
```

训练 Actor 时，以样本为单位随机将整个 `ref_chunk` 置零：

```text
部分样本：Actor 看到 VLA 参考动作
部分样本：Actor 不看到 VLA 参考动作
```

该 dropout：

- 只作用于 Actor 的输入；
- 不删除 BC target 中的 VLA 参考动作；
- 推理时不启用。

目的在于防止 Actor 完全依赖 VLA 动作。当 VLA 参考动作不理想或出现分布偏移时，Actor 仍能利用 `z_rl + proprio` 进行一定程度的修正。

## 11. 人工接管数据

回放逐 step 记录控制来源：

```text
BASE / RL / HUMAN / MIXED
```

人工接管时，系统同时保留：

```text
ref_chunk     ：VLA 当时建议执行的动作
action_chunk  ：人类实际修正后执行的动作
```

训练时，Actor 输入仍然是：

```text
z_rl + proprio + ref_chunk
```

但人工步骤的 BC target 改成实际人工动作。因此人工接管数据相当于提供了：

```text
给定 VLA 建议，应该如何修正动作
```

注意，系统记录的是每个控制 tick 采样到的最新人工动作，而不是把底层遥操作事件流直接写入 replay。

## 12. `critical_phase` 与 `full_task`

### `critical_phase`

回合一开始就进入精细操作段，例如已经接近插接位置，只执行插入或对准。

适用于：

- 聚焦关键操作；
- 减少非关键数据；
- 快速验证 Actor 是否能改善关键动作。

### `full_task`

完整执行任务：

```text
前缀阶段：冻结 VLA 基础策略
      │
      ▼
人工信号 c：标记关键阶段开始
      │
      ▼
关键阶段：切换到关键控制策略
```

非关键前缀不写入 replay，也不使用 Actor。这样可以避免简单的移动阶段产生大量无关数据，稀释关键接触操作的学习信号。

## 13. Replay transition 的内容

每条 replay transition 主要包括：

- `z_rl`、`proprio`；
- 当前 `ref_chunk`；
- 实际执行的 `action_chunk`；
- chunk 内 `rewards`、`done`；
- `next_z_rl`、`next_proprio`、`next_ref_chunk`；
- `source` 与逐 step 的 `source_chunk`；
- `collection_phase`；
- `episode_id`、`step_id`、`success`、`intervention_flag`。

原始轨迹先保存，回合结束后再构建 replay window。这样可以根据完整回合信息处理：

- 人工接管边界；
- 策略重新接管时的 restart anchor；
- 最终 terminal-aligned window；
- 缺失的 Machine A 特征回填。

当 `step_trace_stride: 0` 时，使用动作块边界 replay；当大于 0 时，可根据原始逐 step 轨迹构建更稠密的 replay window。

Replay journal 是追加式 pickle 流。Replay Manager 启动时恢复已有内容，并从最大的 `episode_id + 1` 继续编号。

## 14. Machine A / Machine B 分工

| 模块 | 位置 | 职责 |
|---|---|---|
| 冻结 VLA + RLT | Machine A | 图像/语言理解，生成 `z_rl` 和 `ref_chunk` |
| Actor service | Machine B | 低延迟生成精修动作块 |
| Learner service | Machine B | 从 replay 训练 Actor/Critic |
| Replay Manager | Machine B | 回放缓存、持久化与采样 |
| ROS rollout | 机器人侧 | 读取观测、执行动作、处理复位和人工信号 |

这样划分的原因是：VLA 推理通常较重，Actor 推理需要低延迟，而 Learner 训练不应阻塞机器人控制。

## 15. 训练启动顺序

### 15.1 安装在线 RL 环境

```bash
cd openpi-RLT/rlt_online_rl
conda create -y -n rlt_online_rl310 python=3.10 pip
conda activate rlt_online_rl310
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ../packages/openpi-client
python -m pip install -e .
```

ROS rollout 和键盘客户端运行前需要：

```bash
source /opt/ros/humble/setup.bash
```

### 15.2 启动 Machine B

```bash
cd openpi-RLT/rlt_online_rl
conda activate rlt_online_rl310
python launch/launch_machine_b.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml
```

### 15.3 启动 Machine A

```bash
cd openpi-RLT
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir <checkpoint-dir> \
  --port 8000 \
  --shared-prefix-inference
```

`--shared-prefix-inference` 只是推理延迟优化：复用同一份 VLA prefix 输出和 KV cache，同时生成 `z_rl` 与 `ref_chunk`。它不改变训练、模型权重、checkpoint、归一化和在线 RL payload。

### 15.4 启动机器人 rollout

```bash
cd openpi-RLT/rlt_online_rl
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python launch/launch_robot_rollout.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000
```

再启动训练键盘客户端：

```bash
python keyboard_toggle_teleop_record_reward_isolation.py
```

## 16. 评估流程

评估不启动 Learner 和 Replay，只运行 Actor 推理和机器人 rollout：

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

评估使用确定性的 Actor 均值。键盘客户端：

```bash
python keyboard_actor_eval.py
```

## 17. 一句话总结

> RLT 先把冻结 VLA 的高维视觉-语言状态压缩成适合 RL 使用的 `z_rl`；随后在线 Actor 在 `z_rl + proprio + VLA 参考动作` 的条件下，利用回放、Twin Critic、行为克隆、Q 优化、动作平滑约束以及人工接管数据，学习对 VLA 动作块进行安全而有效的局部修正。
