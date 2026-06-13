# BipedalWalker-v3 Hardcore — SAC 强化学习训练项目

> 使用 **Soft Actor-Critic (SAC)** 算法训练双足机器人在困难地形中行走。  
> 基于 [Gymnasium](https://gymnasium.farama.org/) + [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) + PyTorch CUDA。  
> 经过 **8 轮迭代实验**，从基线（-109）到突破 **+2056**，并在 Round 8 实现 **100 次评估中位数 1972.89** 的稳定表现，实现了接近人类的行走步态。

---

## 项目简介

**环境**: `BipedalWalker-v3` (hardcore=True)  
**算法**: SAC (Soft Actor-Critic)  
**框架**: Stable-Baselines3 2.8.0  
**设备**: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)

困难版在基础双足行走之上，加入了随机生成的台阶、深坑和高低不平的木桩。传统 DDPG 在此环境几乎全军覆没，而 SAC 的最大熵机制和自适应温度系数使其成为该环境的终极试金石。

本项目通过**多轮迭代实验**（8 轮）探索了奖励塑形（Reward Shaping）的演进：
- 从**基线无塑形**（Round 1）
- 到**温和塑形**（Round 2，+527）
- 到**激进增强**（Round 3/4，失败）
- 到**条件化塑形**（Round 5/6，+1705）
- 到**速度挂钩奖励**（Round 7，+2056）
- 到**交替步态奖励**（Round 8，中位数 1972.89，100次评估）

最终实现了**高分 + 人类步态**的双重目标。

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
# 查看第 7 轮摘要
cat ./logs/round_7/training_summary.json

# 批量对比所有轮次
python -c "
import json, glob
for path in sorted(glob.glob('./logs/round_*/training_summary.json')):
    with open(path) as f:
        d = json.load(f)
    print(f'{path}: steps={d[\"training\"][\"total_timesteps\"]}')
"
```

---

## 三种训练模式

脚本支持三种多轮训练模式，通过修改 `train_sac.py` 顶部的两个变量控制：

```python
ROUND_ID = 7        # 轮次编号
RESUME_FROM = "./models/round_6/sac_bipedalwalker_100000_steps.zip"  # 模型加载路径
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
ROUND_ID = 7
RESUME_FROM = "./models/round_7/sac_bipedalwalker_500000_steps.zip"
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
├── log.md                   # 训练日志记录（手动维护，含8轮完整实验记录）
├── requirements.txt         # Python 依赖清单
├── test_env.py             # 环境验证脚本
├── reward_shaping.py       # 奖励塑形包装器（核心创新）
├── train_sac.py            # SAC 训练主脚本
├── eval_model.py           # GUI 评估程序（模型测试、录屏、统计）
├── ppt_replace_font.py     # PPT 字体替换工具（基础版）
└── ppt_replace_font_v2.py  # PPT 字体替换工具（完整版，含 theme 字体）
```

---

## 配置参数说明

编辑 `train_sac.py` 顶部的配置区：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOTAL_TIMESTEPS` | 10,000,000 | 本轮训练总步数 |
| `LEARNING_RATE` | 3e-4 | SAC 学习率 |
| `BATCH_SIZE` | 256 | 每次梯度更新的样本数 |
| `BUFFER_SIZE` | 1,000,000 | 经验回放池容量 |
| `NET_ARCH` | [256, 256] | Actor/Critic 隐藏层结构 |
| `EVAL_FREQ` | 50,000 | 评估频率（步数） |
| `CHECKPOINT_FREQ` | 100,000 | 检查点保存频率 |
| `ROUND_ID` | 9 | 轮次编号（自动创建独立目录） |
| `RESUME_FROM` | `./models/round_8/...` | 预训练模型路径（Round 9） |
| `ENT_COEF` | "auto" | 探索程度：自动调整 |
| `FORWARD_WEIGHT` | 2.5 | 前进速度奖励权重 |
| `UPRIGHT_WEIGHT` | 0.5 | 站立姿态奖励权重 |
| `STALL_PENALTY` | 1.0 | 不动惩罚强度 |
| `STALL_THRESHOLD` | 0.05 | 判定为卡住的水平速度阈值 |
| `LIFT_WEIGHT` | 0.1 | 抬腿跨步奖励权重（速度挂钩） |
| `STRIDE_WEIGHT` | 0.05 | 空中步态奖励权重（速度挂钩） |
| `ALTERNATING_WEIGHT` | 0.5 | 交替步态奖励权重（速度挂钩） |
| `ENABLE_EARLY_TERMINATION` | True | 卡住时是否提前终止 episode |

---

## 奖励塑形 (Reward Shaping) — 核心创新

BipedalWalker-v3 的原始奖励函数存在一个**局部最优陷阱**：机器人发现"半蹲不动"比"走路摔倒"的累积奖励更高，于是选择躺平。这是困难版训练中最常见的问题。

### 演进历程

| 版本 | 核心机制 | 问题 | 结果 |
|------|---------|------|------|
| **基础版** (Round 1) | 无塑形 | 机器人半蹲不动 | 奖励 -109 |
| **温和版** (Round 2) | 前进奖励 + 不动惩罚 | 小碎步行走 | 奖励 +527 |
| **激进版** (Round 3/4) | 高惩罚 + 高奖励 | 策略不稳定，原地踱步 | 奖励 +96 |
| **条件化** (Round 5/6) | 步态奖励只在前进时生效 | 奖励黑客（原地抬腿拿奖励） | 奖励 +1705 |
| **速度挂钩** (Round 7) | 步态奖励与速度成正比 | 蹭行奖励极少，大步快走奖励多 | 奖励 **+2056** |
| **交替步态** (Round 8) | 左右脚交替着地奖励 | 单脚跳不触发，强制真正走路 | 中位数 **1972.89** (100次) |

### 当前交替步态 + 速度挂钩机制

```python
# 抬腿奖励：只在前进时生效，且与速度成正比
if y_velocity > 0.05 and x_velocity > 0.1:
    lift_bonus = LIFT_WEIGHT * y_velocity * x_velocity

# 步态奖励：只在前进时生效，且与速度成正比
if x_velocity > 0.1:
    if 单脚离地:
        stride_bonus = STRIDE_WEIGHT * x_velocity
    elif 双脚离地:
        stride_bonus = STRIDE_WEIGHT * x_velocity * 1.5

# 交替步态奖励：高权重，仅左右脚交替着地时触发，且与速度成正比
if x_velocity > 0.1:
    if (leg1_touched and leg2_lifted) or (leg2_touched and leg1_lifted):
        alternating_bonus = ALTERNATING_WEIGHT * x_velocity
```

**效果**：
- 独脚蹭行（速度 0.1）→ stride_bonus = 0.005, alternating_bonus = 0（极少或不触发）
- 正常走路（速度 0.5）→ stride_bonus = 0.025, alternating_bonus = 0.25（正常）
- 大步快跑（速度 1.0）→ stride_bonus = 0.05, alternating_bonus = 0.50（最多）
- 单脚跳（无交替）→ alternating_bonus = 0（不触发）

### 启用/禁用

在 `train_sac.py` 配置区控制：

```python
# 奖励塑形开关
USE_REWARD_SHAPING = True   # 启用
# USE_REWARD_SHAPING = False  # 禁用（使用原始奖励）

# 塑形参数微调
FORWARD_WEIGHT = 2.5        # 前进奖励权重
STALL_PENALTY = 1.0         # 不动惩罚强度
LIFT_WEIGHT = 0.1           # 抬腿奖励（速度挂钩）
STRIDE_WEIGHT = 0.05        # 步态奖励（速度挂钩）
ALTERNATING_WEIGHT = 0.5    # 交替步态奖励（速度挂钩）
ENABLE_EARLY_TERMINATION = True   # 是否提前终止
```

> **建议**: 困难版强烈建议开启奖励塑形。当前版本使用**交替步态 + 速度挂钩**机制，既能获得高分，又能实现接近人类的行走步态。

### 独立测试

```bash
python reward_shaping.py
```

会运行 500 步随机动作测试，打印每 100 步的原始奖励 vs 塑形奖励对比。

---

## 多轮实验记录

| 轮次 | 起始方式 | 总步数 | 最高奖励 | 最终奖励 | 关键参数 | 结果 |
|------|----------|--------|----------|----------|----------|------|
| Round 1 | 从头 | 49,999 | -48.61 | -109.88 | 无塑形 | 陷入局部最优，半蹲不动 |
| Round 2 | 预训练 + 温和塑形 | 503,978 | +527.78 | +527.78 | FW=2.0, SP=1.0, ST=0.05 | 🎉 成功学会行走 |
| Round 3 | Round 2 + 增强塑形 | 643,052 | +419.74 | +419.74 | FW=5.0, SP=3.0, LR=1e-4 | ⚠️ 策略退化，学习率过低 |
| Round 4 | Round 2 + 恢复LR | 549,999 | +431.44 | +96.40 | FW=5.0, SP=3.0, LR=3e-4 | ⚠️ 增强塑形本身不稳定 |
| Round 5 | Round 2 + 降低塑形 | 462,523 | **+572.51** | +572.51 | FW=2.0, SP=1.0, ST=0.05 | ✅ 历史新高，但小碎步 |
| Round 6 | Round 5 + 修正维度 | ~50,000 | +47.81 | +47.81 | 修正obs维度 | ❌ 修正维度 + 加载旧模型 = 崩溃 |
| Round 6R | 从头 + 精确塑形 | 100,000 | +79.83 | +79.83 | 正确维度 + 步态奖励 | 奖励黑客问题 |
| Round 7 | Round 6R + 条件化 | 1,200,000 | **+2056.50** | **+2056.50** | 速度挂钩 LIFT/STRIDE | 🎉 里程碑！高分 + 人类步态 |
| Round 8 | Round 7 续 + 交替步态 | 1,169,714 | **+2025.46** | **+1976.74** | 交替步态 + 关闭渲染 | ✅ 100次评估中位数 1972.89，稳定 +2000 |
| Round 9 | Round 8 + 最终优化 | ⏳ 进行中 | ⏳ | ⏳ | 延续 Round 8 | 最终作业提交模型 |

> **详细实验记录**: 见 `log.md` — 包含每轮的完整数据、终端输出、失败分析和解决方案。

---

## 关键发现

### 1. 奖励黑客 (Reward Hacking)

无条件步态奖励（`LIFT_WEIGHT`、`STRIDE_WEIGHT`）导致模型发现：
- 原地抬腿也能拿奖励 → "奖励黑客"
- 单脚离地蹭行也能拿奖励 → 不像人类走路

**解决**：将步态奖励改为**条件化**（只在前进时生效）+ **速度挂钩**（速度越快奖励越多）。

### 2. Obs 维度修正的陷阱

Round 2~5 的模型在"错误"维度上（`obs[1]` 角速度而非 `obs[2]` 前进速度）成功训练到了 +572。

**教训**：修正 obs 维度后**不能加载旧模型**，必须从零开始训练。旧模型已将错误的特征表示编码进网络权重。

### 3. 策略顿悟 (Policy Breakthrough)

SAC 训练的典型曲线：长期探索（0~40万步波动剧烈）→ 突然突破（40万步后奖励飙升）。

Round 7 在 40~50万步时从 +410 飙升到 +1705，印证了 BipedalWalker 的"顿悟"模式。

### 4. 人类步态 vs 高分

最初认为"高分 = 好策略"，但发现 +1705 的蹭行策略虽然高分，却不像人类。

最终通过速度挂钩奖励实现了**+2056 高分 + 接近人类步态**的双重目标。

---

## 最佳成绩

| 指标 | 数值 | 步数/轮次 |
|------|------|----------|
| **最高单次评估奖励** | **+2056.50** | 1,200,000 (Round 7) |
| **100 次评估中位数** | **1972.89** | 1,169,714 (Round 8) |
| 100 次评估均值 | 1935.61 ± 81.98 | Round 8 独立测试 |
| 标准差（最佳） | 0.59 | 300,000 (Round 8) |
| Episode 长度 | 1108.50 ± 75.50 | 稳定 |
| 步态 | 接近人类（交替步态，自然抬腿） | 交替步态 + 速度挂钩 |
| 学习率 | 3e-4 | 未改变 |
| 网络结构 | [256, 256] | 未改变 |

---

## 训练难点备忘

1. **超参数敏感**: 网络层数、Batch Size、经验回放池容量设置不合理时，机器人会在原地疯狂前空翻或直接趴下
2. **训练时间长**: 通常需要 50万~200万步才能看到机器人跌跌撞撞跨越障碍
3. **地形随机**: 每次 episode 的台阶、深坑、木桩位置随机生成，要求策略具有强泛化能力
4. **SAC 优势**: 最大熵机制和自适应温度系数 α 使其在新地形上比 DDPG 更不容易卡死
5. **渲染开销**: `render=True` 弹出 pygame 窗口时会暂停训练，仅用于观察，正式跑数据建议关闭
6. **局部最优陷阱**: 机器人容易学会"半蹲不动"——因为原始奖励中摔倒惩罚 (-100) 远大于存活惩罚 (-0.00035/步)。必须通过奖励塑形打破
7. **奖励黑客**: 步态奖励可能被滥用，需要条件化 + 速度挂钩才能引导出人类步态

---

## 常见问题

### Q: 训练时弹出 pygame 窗口很卡？
A: 这是正常的。评估渲染会暂停训练。如果不需要观看，将 `train_sac.py` 中 `eval_env` 的 `render_mode="human"` 改回 `None`，并将 `EvalCallback` 的 `render=True` 改回 `False`。

### Q: 按 Ctrl+C 中断后模型会丢失吗？
A: 不会。脚本有 `try/finally` 保护，中断时会自动保存 `sac_bipedalwalker_final.zip` 和 JSON 摘要。

### Q: 如何对比多轮实验结果？
A: 每轮有独立的 `training_summary.json`，可用 Python 脚本批量读取对比：
```python
import json, glob
for path in sorted(glob.glob('./logs/round_*/training_summary.json')):
    with open(path) as f:
        d = json.load(f)
    print(f'{path}: steps={d["training"]["total_timesteps"]}')
```

### Q: 可以加载其他算法训练的模型吗？
A: 不可以。SB3 的 `.zip` 文件包含算法特定状态，只能由同算法加载。

### Q: 为什么步态奖励需要速度挂钩？
A: 固定步态奖励会导致"奖励黑客"——模型发现原地抬腿或单脚蹭行也能拿奖励。速度挂钩后，只有真正快走时才能获得丰厚奖励，从而引导出自然步态。

---

## 辅助工具

### GUI 评估程序 (`eval_model.py`)

`eval_model.py` 是一个基于 tkinter + pygame 的 GUI 评估程序，用于可视化测试训练好的模型步态。

**功能**：
- 模型选择：选择任意 `.zip` 模型文件
- 批量评估：可设置评估次数（1~100 次）
- 奖励塑形开关：可选择是否启用奖励塑形进行对比
- 实时渲染：pygame 窗口实时显示机器人行走
- 录屏保存：支持将评估过程保存为视频文件
- 统计报告：评估结束后显示均值、标准差、中位数、最大/最小值

**启动方式**：
```bash
python eval_model.py
```

**界面说明**：
- **模型路径**：点击 "Browse" 选择 `./models/round_8/sac_bipedalwalker_final.zip`
- **评估次数**：输入评估次数（推荐 10~100 次）
- **启用奖励塑形**：勾选后使用与训练时相同的奖励塑形参数
- **开始评估**：点击后弹出 pygame 窗口，实时显示机器人行走
- **保存视频**：勾选后自动保存录屏到当前目录

**评估结果示例**：
```
==================================================
评估完成！
  平均奖励: 1935.61 ± 81.98
  中位数:   1972.89
  最小:     1772.83
  最大:     1994.07
  平均步长: 1033.4
==================================================
```

**注意**：Windows 上 pygame 实时渲染窗口容易崩溃，训练时建议关闭渲染（`render=False`），评估时单独使用此程序观察步态。

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `train_sac.py` | 主训练脚本，支持三种多轮训练模式 + 奖励塑形 |
| `reward_shaping.py` | 奖励塑形包装器，核心创新：速度挂钩步态奖励 |
| `eval_model.py` | GUI 评估程序，支持模型选择、批量评估、实时渲染、录屏 |
| `test_env.py` | 环境验证，确认 Box2D 和 Gymnasium 正常工作 |
| `log.md` | 手动维护的详细训练日志，记录8轮完整实验 |
| `requirements.txt` | Python 包依赖清单 |

---

## 参考

- [Gymnasium BipedalWalker 文档](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)
- [Stable-Baselines3 SAC 文档](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [SAC 论文 (Haarnoja et al., 2018)](https://arxiv.org/abs/1812.05905)
