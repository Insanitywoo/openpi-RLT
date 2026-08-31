# openpi-RLT

`openpi-RLT` 是一个基于 openpi 的 **RL Token（RLT）** 复现项目，面向真实机器人的在线强化学习。它保留了上游 [openpi](https://github.com/Physical-Intelligence/openpi) 的 VLA 训练与推理栈，并补充了端到端运行 RLT 所需的 RL-token 模块、策略服务链路、回放运行时、Actor-Critic 学习器以及真实机器人 rollout 工具。

据我们所知，openpi-RLT 是首个开源的、基于 openpi/pi0.5 的真实机器人 RLT 风格流水线复现。项目已在以太网线插接任务上演示：RL-token 适配、冻结 VLA 参考策略服务、在线 Actor-Critic 学习、经验回放、rollout 和评估。

## 真实机器人结果

第一个演示是冻结的 VLA 基线；第二个演示是经过在线训练后的 RLT 策略。

<p align="center"><strong>冻结的 VLA 基线</strong></p>

<p align="center">
  <img alt="以太网插接任务上的冻结 VLA 基线" src="media/ethernet_vla_baseline.gif" width="48%"/>
</p>

<p align="center"><strong>RLT 策略</strong></p>

<p align="center">
  <img alt="以太网插接任务上的 RLT 策略" src="media/ethernet_rlt_policy.gif" width="48%"/>
</p>

同时提供了高分辨率 MP4：
[VLA 基线](media/ethernet_vla_baseline.mp4) 与
[RLT 策略](media/ethernet_rlt_policy.mp4)。

## 核心组件

项目保留了上游 openpi 的目录结构。下表列出构成 openpi-RLT 的主要 RLT 专属新增内容和修改过的入口。

| 组件 | 路径 | 作用 |
| --- | --- | --- |
| RL-token 模型集成 | `src/openpi/models/rl_token.py`、`src/openpi/models/pi0.py` | 增加 RL-token 编码器/解码器，以及供 RLT 策略使用的 pi0/pi0.5 接入点。 |
| 训练入口 | `src/openpi/training/config.py`、`scripts/train_rlt.py` | 注册 RLT 配置，并启动第一阶段的 RL-token 训练。 |
| 远程策略服务 | `scripts/serve_rlt_policy.py`、`packages/openpi-client/` | 向在线 RL 运行时提供冻结 VLA 参考信息与紧凑的 RLT 特征。 |
| 部署策略适配器 | `src/openpi/policies/agilexbag_image_policy.py` | 将图像观测和动作块适配为部署时使用的格式。 |
| 在线 RL 运行时 | `rlt_online_rl/src/rlt_online_rl/` | 包含 actor、critic、学习器、回放、推理和 rollout 端运行时模块。 |
| 实验启动与配置 | `rlt_online_rl/launch/`、`rlt_online_rl/configs/` | 为以太网插接等实验提供启动脚本和运行时配置。 |
| 机器人接口桥接 | `rlt_online_rl/train_deploy_alignment/` | 将在线 RL 运行时与真实机器人控制和信号接口连接起来。 |
| 回放与分析工具 | `rlt_online_rl/scripts/` | 包含离线回放检查、回放导出和实验辅助工具。 |
| 演示资源与说明 | `media/`、`docs/` | 存放 README 演示素材与专项配置说明。 |

详细的运行时说明见 [rlt_online_rl/README.md](rlt_online_rl/README.md)；软件包内部结构概览见 [rlt_online_rl/src/rlt_online_rl/README.md](rlt_online_rl/src/rlt_online_rl/README.md)。

## 快速开始

从仓库根目录克隆并安装 openpi/RLT 栈：

```bash
git clone https://github.com/Yyshadow/openpi-RLT.git
cd openpi-RLT
uv sync
uv pip install -e .
```

RLT 训练配置注册在 [`src/openpi/training/config.py`](src/openpi/training/config.py) 中。典型的第一阶段训练命令如下：

```bash
uv run scripts/train_rlt.py rlt_pi05_agilexbag_image_delta_joint \
  --exp-name <run-name> \
  --overwrite
```

获得训练好的 checkpoint 后，可通过以下命令提供冻结的 VLA/RLT 策略服务：

```bash
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir <checkpoint-dir> \
  --port 8000
```

关于真实机器人在线 RL 的启动顺序、键盘控制、回放语义和仅评估 rollout 流程，请参阅 [rlt_online_rl/README.md](rlt_online_rl/README.md)。

## 与上游工作的关系

本项目遵循论文 **RL Token: Bootstrapping Online RL with Vision-Language-Action Models** 中的 RLT 方法，并以 openpi 的 pi0.5 作为基础 VLA 栈。本 fork 中与 RLT 有关的改动叠加在上游 openpi 之上，而不是以另一套骨干网络替换它。

主要实现包括：

- RL-token 编码器/解码器模块及其 pi0.5 集成；
- RL-token 训练，并可选地加入监督式 VLA 微调项；
- 冻结 VLA 策略服务：返回参考动作块和紧凑的 RL-token 特征；
- 轻量级在线 Actor-Critic 运行时：在动作块级别进行动作修正；
- 真实机器人回放采集、回合收尾、人工干预处理和仅评估 rollout 支持。

## 贡献者

核心贡献者按贡献顺序列出：

Yi Yang<sup>&#42;</sup>, Huaihang Zheng<sup>&#42;</sup>, Kai Ma<sup>&#42;&#8224;</sup>, Tian Xie, Guozheng Li, Shenglin Xu,
Xiangyu Wang, Yiren Ma, Baoxu Liu<sup>&#8224;</sup>

<sub>&#42; 共同第一作者。&#8224; 项目负责人。</sub>

## 引用

若您在工作中使用了本代码，请同时引用本仓库、原始 RLT 论文和 openpi。

```bibtex
@misc{openpi_rlt_2026,
  title = {openpi-RLT: Real-Robot RLT Reproduction on openpi},
  author = {Yi Yang and Huaihang Zheng and Kai Ma and Tian Xie and Guozheng Li and Shenglin Xu and Xiangyu Wang and Yiren Ma and Baoxu Liu},
  year = {2026},
  note = {Open-source real-robot reproduction of RL Token (RLT) built on openpi.}
}
```

## 致谢

本仓库基于 Physical Intelligence 的 [openpi](https://github.com/Physical-Intelligence/openpi)，并遵循 RLT 论文 [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://pi.website/research/rlt) 的方法。

## 许可证

请参阅 [LICENSE](LICENSE) 和 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。
