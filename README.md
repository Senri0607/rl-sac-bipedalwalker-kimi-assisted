# BipedalWalker-v3 Hardcore — SAC 强化学习训练项目

> 使用 **Soft Actor-Critic (SAC)** 算法训练双足机器人在困难地形中行走。  
> 基于 [Gymnasium](https://gymnasium.farama.org/) + [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) + PyTorch CUDA。

---

## 项目简介

**环境**: `BipedalWalker-v3` (hardcore=True)  
**算法**: SAC (Soft Actor-Critic)  
**框架**: Stable-Baselines3  
**设备**: NVIDIA GPU (CUDA 12.6)

困难版在基础双足行走之上，加入了随机生成的台阶、深坑和高低不平的木桩。传统 DDPG 在此环境几乎全军覆没，而 SAC 的最大熵机制和自适应温度系数使其成为该环境的终极试金石。

---

## 环境要求

| 项目 | 版本/要求 |
|------|----------|
| Python | 3.13+ |
| PyTorch | 2.12.0+cu126 (CUDA 12.6) |
| Gymnasium | 1.2.3 |
| Stable-Baselines3 | 2.8.0 |
| Box2D | 2.3.8 (需从源码编译) |
| pygame | 2.6.1 |
| 操作系统 | Windows 10/11 |
| GPU | NVIDIA (推荐 RTX 4060 或更高) |
| 编译工具 | Microsoft C++ Build Tools |

---

## 安装步骤

### 1. 安装 Microsoft C++ Build Tools

Box2D 需要从源码编译 C++ 扩展，必须安装 Build Tools：

1. 访问 https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 下载并运行 **Build Tools for Visual Studio**
3. 安装时勾选 **"使用 C++ 的桌面开发"** 工作负载
4. 等待安装完成

### 2. 安装 Python 依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install gymnasium stable-baselines3 tensorboard matplotlib numpy tqdm rich pygame
```

### 3. 安装 Box2D (从源码编译)

由于 Python 3.13 暂无预编译 wheel，需要从源码安装：

```bash
# 下载源码包
pip download box2d-py --no-deps -d .
tar -xzf box2d-py-2.3.8.tar.gz

# 修改 setup.py 使用 python -m swig 代替系统 swig
# （详见项目中的 box2d-py-2.3.8/setup.py 修改说明）

# 设置 MSVC 环境后编译安装
python setup.py install
```

### 4. 验证环境

```bash
python test_env.py
```

输出 `[PASS] 环境测试全部通过！` 即表示就绪。

---

## 快速开始

### 启动训练

```bash
python train_sac.py
```

### 监控训练

```bash
# TensorBoard 实时曲线
tensorboard --logdir=./logs

# 在浏览器打开 http://localhost:6006
```

### 查看训练摘要

每轮训练结束后自动生成 JSON 摘要：

```bash
# 查看第 1 轮摘要
cat ./logs/round_1/training_summary.json
```

---

## 三种训练模式

脚本支持三种多轮训练模式，通过修改 `train_sac.py` 顶部的两个变量控制：

```python
ROUND_ID = 1        # 轮次编号
RESUME_FROM = None  # 模型加载路径
```

### 模式 A：完全从头（基线实验）

```python
ROUND_ID = 1
RESUME_FROM = None
```

- 随机初始化网络权重
- 全新经验回放池
- 输出到 `./logs/round_1/` 和 `./models/round_1/`

### 模式 B：加载预训练当新轮次起点

```python
ROUND_ID = 2
RESUME_FROM = "./models/round_1/sac_bipedalwalker_final.zip"
```

- 加载上一轮训练好的 Actor/Critic 网络权重
- 经验回放池清空，重新探索
- 可修改超参数（如学习率、网络结构）
- 适用于：换参数继续优化、对比实验

### 模式 C：断点续训（同一轮次接着跑）

```python
ROUND_ID = 1
RESUME_FROM = "./models/round_1/sac_bipedalwalker_500000_steps.zip"
```

- 从指定检查点恢复模型权重
- 继续在同一目录下训练
- 适用于：意外中断后恢复、分阶段训练

> **注意**: SB3 的 `SAC.load()` 不会保存经验回放池，续训时回放池为空。这是框架限制，不影响策略网络本身。

---

## 目录结构

```
.
├── .vscode/
│   ├── launch.json          # VSCode 调试配置 (SAC训练 / 环境测试)
│   └── settings.json        # VSCode 工作区设置
├── logs/
│   └── round_{N}/           # 每轮独立的日志目录
│       ├── training_summary.json   # JSON 训练摘要
│       ├── evaluations.npz       # 评估数据
│       ├── training_log.txt      # 文本训练日志
│       └── SAC_1/                # TensorBoard 事件文件
├── models/
│   └── round_{N}/           # 每轮独立的模型目录
│       ├── sac_bipedalwalker_{steps}_steps.zip   # 检查点
│       ├── sac_bipedalwalker_final.zip           # 最终模型
│       └── best_model.zip                        # 最佳模型
├── README.md                # 本文件
├── log.md                   # 训练日志记录（手动维护）
├── requirements.txt         # Python 依赖清单
├── test_env.py             # 环境验证脚本
└── train_sac.py            # SAC 训练主脚本
```

---

## 配置参数说明

编辑 `train_sac.py` 顶部的配置区：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOTAL_TIMESTEPS` | 10,000,000 | 本轮训练总步数 |
| `LEARNING_RATE` | 3e-4 (Round 1/2) / **1e-4 (Round 3+)** | SAC 学习率 |
| `BATCH_SIZE` | 256 | 每次梯度更新的样本数 |
| `BUFFER_SIZE` | 1,000,000 | 经验回放池容量 |
| `NET_ARCH` | [256, 256] (Round 1/2) / **[512, 512] (Round 3+)** | Actor/Critic 隐藏层结构 |
| `EVAL_FREQ` | 50,000 | 评估频率（步数） |
| `CHECKPOINT_FREQ` | 100,000 | 检查点保存频率 |
| `ROUND_ID` | 1 | 轮次编号（自动创建独立目录） |
| `RESUME_FROM` | None | 预训练模型路径 |
| `ENT_COEF` | "auto" (Round 1/2) / **0.1 (Round 3+)** | 探索程度：自动调整或固定值 |
| `STALL_PENALTY` | 1.0 (Round 2) / **3.0 (Round 3+)** | 不动惩罚强度 |
| `FORWARD_WEIGHT` | 2.0 (Round 2) / **5.0 (Round 3+)** | 前进速度奖励权重 |
| `STALL_THRESHOLD` | 0.05 (Round 2) / **0.15 (Round 3+)** | 判定为卡住的水平速度阈值 |

---

## 奖励塑形 (Reward Shaping)

BipedalWalker-v3 的原始奖励函数存在一个**局部最优陷阱**：机器人发现"半蹲不动"比"走路摔倒"的累积奖励更高，于是选择躺平。这是困难版训练中最常见的问题。

### 解决方案

项目内置了 `reward_shaping.py` 奖励塑形包装器，通过以下机制打破局部最优：

| 塑形项 | 作用 | Round 2 权重 | Round 3+ 权重 |
|--------|------|-------------|--------------|
| **前进速度奖励** | 与 torso 水平速度成正比，走得越快奖励越高 | `forward_weight=2.0` | **`forward_weight=5.0`** |
| **站立姿态奖励** | 躯干角度越接近垂直，奖励越高 | `upright_weight=0.5` | `upright_weight=0.5` |
| **不动惩罚** | 当水平速度低于阈值时给予负奖励，打破"卡住" | `stall_penalty=1.0` | **`stall_penalty=3.0`** |
| **动作平滑奖励** | 惩罚动作剧烈抖动，鼓励连贯行走 | `smooth_weight=0.1` | `smooth_weight=0.1` |
| **提前终止** | 连续卡住超过阈值步数强制结束 episode | `max_stall_steps=100` | `max_stall_steps=100` |
| **卡住判定阈值** | 低于此速度即判定为"卡住" | `stall_threshold=0.05` | **`stall_threshold=0.15`** |

### 启用/禁用

在 `train_sac.py` 配置区控制：

```python
# 奖励塑形开关
USE_REWARD_SHAPING = True   # 启用
# USE_REWARD_SHAPING = False  # 禁用（使用原始奖励）

# 塑形参数微调
FORWARD_WEIGHT = 2.0        # 前进奖励权重
STALL_PENALTY = 1.0         # 不动惩罚强度
ENABLE_EARLY_TERMINATION = True   # 是否提前终止
```

> **建议**: 困难版强烈建议开启奖励塑形。Round 2 使用 `STALL_PENALTY=1.0` 成功打破"躺平"；Round 3 进一步增强到 `STALL_PENALTY=3.0` + `FORWARD_WEIGHT=5.0` + `STALL_THRESHOLD=0.15`，彻底消除小碎步。

### 多轮训练参数演进

| 轮次 | 网络结构 | 学习率 | ENT_COEF | STALL_PENALTY | FORWARD_WEIGHT | STALL_THRESHOLD | 效果 |
|------|----------|--------|----------|---------------|----------------|-----------------|------|
| Round 1 | [256, 256] | 3e-4 | "auto" | ❌ 未启用 | ❌ 未启用 | — | 陷入局部最优，半蹲不动 |
| Round 2 | [256, 256] | 3e-4 | "auto" | 1.0 | 2.0 | 0.05 | 🎉 学会行走，奖励 +527.78 |
| Round 3 | **[512, 512]** | **1e-4** | **0.1** | **3.0** | **5.0** | **0.15** | ⏳ 目标：消除小碎步，走得更快更稳 |

### 独立测试

```bash
python reward_shaping.py
```

会运行 500 步随机动作测试，打印每 100 步的原始奖励 vs 塑形奖励对比。

---

## 训练难点备忘

1. **超参数敏感**: 网络层数、Batch Size、经验回放池容量设置不合理时，机器人会在原地疯狂前空翻或直接趴下
2. **训练时间长**: 通常需要 500万~1000万步以上才能看到机器人跌跌撞撞跨越障碍
3. **地形随机**: 每次 episode 的台阶、深坑、木桩位置随机生成，要求策略具有强泛化能力
4. **SAC 优势**: 最大熵机制和自适应温度系数 α 使其在新地形上比 DDPG 更不容易卡死

---

## 常见问题

### Q: 训练时弹出 pygame 窗口很卡？
A: 这是正常的。评估渲染会暂停训练。如果不需要观看，将 `train_sac.py` 中 `eval_env` 的 `render_mode="human"` 改回 `None`，并将 `EvalCallback` 的 `render=True` 改回 `False`。

### Q: 按 Ctrl+C 中断后模型会丢失吗？
A: 不会。脚本有 `try/finally` 保护，中断时会自动保存 `sac_bipedalwalker_final.zip` 和 JSON 摘要。

### Q: 如何对比多轮实验结果？
A: 每轮有独立的 `training_summary.json`，可用 Python 脚本批量读取对比：
```python
import json
glob("./logs/round_*/training_summary.json")
```

### Q: 可以加载其他算法训练的模型吗？
A: 不可以。SB3 的 `.zip` 文件包含算法特定状态，只能由同算法加载。

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `train_sac.py` | 主训练脚本，支持三种多轮训练模式 + 奖励塑形 |
| `reward_shaping.py` | 奖励塑形包装器，解决机器人"卡住不动"问题 |
| `test_env.py` | 环境验证，确认 Box2D 和 Gymnasium 正常工作 |
| `log.md` | 手动维护的训练日志，记录每轮关键结果 |
| `requirements.txt` | Python 包依赖清单 |

---

## 参考

- [Gymnasium BipedalWalker 文档](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)
- [Stable-Baselines3 SAC 文档](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [SAC 论文 (Haarnoja et al., 2018)](https://arxiv.org/abs/1812.05905)
