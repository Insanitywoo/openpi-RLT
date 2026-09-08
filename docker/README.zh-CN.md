# OpenPI-RLT 镜像与分布式运行

`Dockerfile.rlt-runtime` 固化两个解释器环境：

- `/.venv`：Python 3.11 OpenPI、Pi0、RLT Stage 1、Machine A；
- `/online-rl-venv`：Python 3.10 Online RL、Gym-ALOHA 仿真、Machine B。

数据、Pi0 权重、RLT checkpoint、replay 和日志不烘焙进镜像，通过
`OPENPI_CFS_ROOT` 挂载 CFS。这使同一镜像可以在单卡和多卡节点复用。

构建与导出：

```bash
docker build -f docker/Dockerfile.rlt-runtime -t openpi-rlt:sim-ready .
docker save openpi-rlt:sim-ready | gzip > openpi-rlt-sim-ready.tar.gz
```

单节点启动：

```bash
OPENPI_CFS_ROOT=/mnt/cfs/usr/wujh/openpi-RLT \
  docker compose -f docker/compose.rlt-sim.yaml run --rm rlt_runtime
```

多卡训练由调度系统为容器分配 GPU，再通过项目现有的 `--fsdp-devices N`
参数运行 `scripts/train_rlt.py`。镜像只固定软件，不把节点地址、SSH 私钥、
模型权重或 checkpoint 写入镜像。
