# AGENTS.md — openpi-RLT 项目协作与学习执行约定

## 1. 项目定位

`openpi-RLT` 是在上游 openpi/pi0.5 视觉-语言-动作（VLA）栈上复现 RL Token（RLT）的项目，目标是支持真实机器人在线强化学习。

核心理念：冻结 VLA 提供任务理解和参考动作；RLT 将 VLA prefix 压缩为 `z_rl`；在线轻量 Actor-Critic 根据 `z_rl + proprio + ref_chunk` 学习局部动作修正。

```text
Machine A：冻结 VLA + RLT
  输出 z_rl 与 ref_chunk

Machine B：Actor service + Learner service + Replay manager
  Actor 精修 VLA 参考动作块

执行端：ROS 真实机器人或后续 ManiSkill 仿真 adapter
```

## 2. 核心目录

```text
src/openpi/models/rl_token.py       RLT Encoder/Decoder
scripts/train_rlt.py                阶段 1 RLT 训练入口
scripts/serve_rlt_policy.py         Machine A：VLA/RLT WebSocket 服务
rlt_online_rl/src/rlt_online_rl/    Actor/Critic/Replay/Learner/EnvDriver
rlt_online_rl/launch/               Machine B、rollout、评估启动器
rlt_online_rl/configs/              在线 RL 配置
rlt_online_rl/train_deploy_alignment/ ROS 与真实机器人桥接
```

## 3. 核心算法约定

### 阶段 1：RL-token

```text
VLA prefix embedding
  → RLTokenEncoder
  → RL tokens / z_rl
  → RLTokenDecoder
  → reconstruction loss
```

```text
L_RLT = MSE(reconstructed_prefix, stop_gradient(vla_prefix))
L_total = L_RLT + alpha * L_VLA  # alpha > 0 时
```

### 阶段 2：在线 RL

Actor 输入：`z_rl`、`proprio`、`ref_chunk`。  
Actor 输出：精修后的动作块。  
Critic 输入：`z_rl`、`proprio`、`action_chunk`。

```text
actor_loss = bc_weight * bc_penalty
           - q_weight * actor_q
           + delta_weight * delta_penalty
```

BC target：

- `BASE` / `RL` step：使用 VLA `ref_chunk`；
- `HUMAN` / `MIXED` step：使用实际执行的 `action_chunk`。

关键语义：Actor 是“VLA 动作编辑器”，不是脱离 VLA 从零学习的策略。训练时可使用 reference dropout；推理时不使用。

## 4. 当前复现路线

完整计划见仓库根目录：[REPRODUCTION_PLAN.zh-CN.md](REPRODUCTION_PLAN.zh-CN.md)。

当前路线：

1. 本地用 uv 建立根项目和在线 RL 环境，完成软件链路验证；
2. 跑通轻量测试并记录本机 GPU 兼容性限制；
3. 用 fake/synthetic 数据验证 RLT 阶段 1；
4. 将已验证代码同步到百舸云 Pro6000 开发机；
5. 在云端完成完整测试、模型下载和正式训练；
6. 验证 Machine A 和 Machine B；
7. 新增 ManiSkill adapter；
8. 在仿真中跑通 warmup、replay、learner、snapshot 闭环；
9. 之后才规划 ROS/AgileX 真实机器人。

首轮验收是**软件链路全通和算法单测**，不是立即复现真实 Ethernet insertion 成功率。

## 5. 用户学习优先：交互式执行规则

用户希望亲自学习和执行。除非用户明确要求自动执行，否则代理必须：

1. 先解释当前步骤在全局链路中的作用；
2. 解释原理、命令参数和风险；
3. 给用户一组最小且可复制的命令；
4. 说明预期输出、成功判据和常见错误；
5. 等待用户亲自运行并贴回输出；
6. 解读输出后再给下一步。

默认**不得自动**：

- 写入或修改代码、配置、文档；
- 安装、更新、卸载依赖；
- 下载模型、数据或镜像；
- 创建/修改云端开发机；
- 启动训练、服务、仿真或后台进程；
- 执行网络、SSH、上传下载；
- 控制真实机器人、ROS 或任何物理执行器。

只有用户明确指示“自动执行/直接执行/帮我运行/帮我修改”某项具体工作时，才可执行对应动作；仍须遵循安全、权限、网络和破坏性操作的确认要求。

## 6. 运行环境与兼容性

- 根 openpi/RLT 项目：Python `>=3.11`，使用 `uv`；
- `rlt_online_rl`：Python `>=3.10,<3.11`，使用独立的 uv 虚拟环境 `rlt_online_rl/.venv`；
- 真实机器人相关：ROS2 Humble；
- 正式训练目标：百舸云 B300 单卡起步，后续按需要扩展到多 GPU；
- 本机 RTX 5070 Ti：轻量验证和开发，不作为正式大模型训练基线。

不要混用两个 Python 环境。涉及 GPU 时优先显式设置进程 GPU 分配，并记录驱动、JAX 和 CUDA 信息。

## 7. 文档顺序

1. `README.zh-CN.md`：项目概览；
2. `RLT_THEORY_REFERENCE.zh-CN.md`：RLT 理论与算法；
3. `rlt_online_rl/README.zh-CN.md`：在线运行时说明；
4. `REPRODUCTION_PLAN.zh-CN.md`：长期执行计划和阶段验收。

## 8. 云端开发机访问约定

本项目的正式训练与完整运行环境位于百舸云开发机；本地主要用于源码阅读、代码修改、文档和 Git 管理。当前本机已配置 SSH 别名：

```text
openpi-rlt
```

代理需要在云端执行环境相关工作时，默认通过 SSH 使用该别名访问远程开发机：

```bash
ssh openpi-rlt '<远程命令>'
```

远程项目工作区约定为（代码与大文件分离）：

```text
/mnt/cfs/usr/wujh/openpi-RLT/repo/openpi-RLT
```

云端个人实验根目录为：

```text
/mnt/cfs/usr/wujh/openpi-RLT
```

其中 `repo/` 放代码，数据、模型、checkpoint、日志、replay、实验输出和可复用下载缓存放在对应的 CFS 持久化目录中。

**系统盘与 CFS 分工是硬规则：**

- 系统盘（`/root/workspace`）放运行时软件、`git`、`tmux`、uv 下载的 Python、两个 `.venv`、`site-packages` 以及临时编译产物；
- CFS（`/mnt/cfs/usr/wujh/openpi-RLT`）放代码、数据集、预训练权重、checkpoint、replay、日志、实验结果和 Hugging Face/W&B 等大文件缓存；
- 不在 CFS 创建 Python 虚拟环境；不把长期资产放在系统盘；
- `/root/workspace` 可能随开发机重建而丢失，只能保存可重建的环境；CFS 是长期持久化位置。

推荐系统盘环境位置：

```text
/root/workspace/openpi-rlt-system/openpi311/.venv
/root/workspace/openpi-rlt-system/online-rl310/.venv
```

推荐 CFS 环境变量：

```bash
export OPENPI_CFS_ROOT=/mnt/cfs/usr/wujh/openpi-RLT
export OPENPI_REPO="$OPENPI_CFS_ROOT/repo/openpi-RLT"
export HF_HOME="$OPENPI_CFS_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$OPENPI_CFS_ROOT/cache"
export WANDB_DIR="$OPENPI_CFS_ROOT/cache/wandb"
export UV_CACHE_DIR="$OPENPI_CFS_ROOT/cache/uv"
```

这些变量只是在当前 shell 中给程序提供“代码、数据和缓存应该去哪”的路径，不会自动创建目录，也不会安装依赖。需要持久化时，应将它们写入系统盘上的个人 shell 配置或项目启动脚本，而不是依赖临时手工输入。`/root/workspace` 仅作为容器临时工作区。

云端操作默认分工：

- 本地：源码分析、跨环境代码修改、文档维护、Git commit/push；
- 云端：GPU/磁盘检查、uv 环境、依赖安装、模型和数据下载、完整测试、训练、服务与仿真。

当用户说“在云端”“开发机上”“云端训练”或“查看云端状态”时，默认理解为通过 `ssh openpi-rlt` 执行；不要求云端安装第二份 Codex。若云端项目尚未 clone，先通过 Git 将本地已推送提交同步到云端。

安全约定：

- SSH 私钥只保留在本机 `~/.ssh/`，不得写入项目或提交 Git；
- SSH 连接参数和私钥路径不得写入公开仓库；项目只记录别名 `openpi-rlt`；
- 长时间命令使用 `tmux` 或 `nohup`，并把日志写入持久化目录；
- 删除 checkpoint、清理 replay、使用 `--overwrite`、停止已有训练或启动高成本长任务前，必须向用户说明影响；
- Machine A、Actor、Replay 初期只监听 `127.0.0.1`，不默认暴露公网端口。
