# 训练日志 — BipedalWalker-v3 Hardcore SAC

> **项目**: 双足机器人困难版 (BipedalWalker-v3 Hardcore)  
> **算法**: SAC (Soft Actor-Critic)  
> **创建时间**: 2026-06-11  
> **训练设备**: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)

---

## 环境信息

| 项目 | 详情 |
|------|------|
| 环境名称 | `BipedalWalker-v3` |
| 困难模式 | `hardcore=True` |
| 观测空间 | 24维连续值 |
| 动作空间 | 4维连续值 (关节扭矩) |
| 算法 | SAC (柔性 Actor-Critic) |
| 框架 | Stable-Baselines3 2.8.0 |
| PyTorch | 2.12.0+cu126 |
| Gymnasium | 1.2.3 |
| Box2D | 2.3.8 (源码编译) |

---

## 全局超参数配置

```python
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
TAU = 0.005
GAMMA = 0.99
ENT_COEF = "auto"
NET_ARCH = [256, 256]
```

## 奖励塑形配置

```python
USE_REWARD_SHAPING = True
FORWARD_WEIGHT = 2.0
UPRIGHT_WEIGHT = 0.5
STALL_PENALTY = 1.0
SMOOTH_WEIGHT = 0.1
ENABLE_EARLY_TERMINATION = True
STALL_THRESHOLD = 0.05
MAX_STALL_STEPS = 100
```

> 奖励塑形用于解决机器人"半蹲不动"的局部最优问题。  
> 如果某轮训练出现卡住现象，可尝试增大 `STALL_PENALTY`（如改为 2.0 或 3.0）。

---

## 训练记录

### Round 1 — 基线实验 (从头训练，无奖励塑形)

| 项目 | 内容 |
|------|------|
| 轮次 | Round 1 |
| 训练步数 | 49,999 / 10,000,000 (仅完成 0.5%) |
| 起始时间 | 2026-06-11 21:11:33 |
| 结束时间 | 2026-06-11 21:19:52 |
| 状态 | ⚠️ **用户中断** |
| RESUME_FROM | `None` (全新初始化) |
| 奖励塑形 | ❌ 未启用 |
| 训练耗时 | 499.04 秒 (约 8.3 分钟) |
| 平均 FPS | ~120 |

**训练统计**:
- 总 Episode 数: **96**
- 平均奖励: **-109.88**
- 最高奖励: **-48.61** (Step 43,459, Length 1600)
- 最低奖励: **-171.33** (Step 46,241, Length 1402)
- 标准差: **19.52**
- 平均 Episode 长度: **511.77**

**关键观察**:

| 阶段 | 步数范围 | Episode 特征 | 现象 |
|------|----------|-------------|------|
| 初期 | 0 ~ 17,000 | 长短不一 (44 ~ 1600 步) | 探索阶段，频繁摔倒 |
| 中期 | 17,000 ~ 30,000 | 大量 1600 步长 episode | **出现"卡住不动"策略**，靠超时存活 |
| 后期 | 30,000 ~ 50,000 | 1600 步与短 episode 交替 | 局部最优固化，偶尔摔倒 |

**评估记录**:
- 本轮未触发评估 (评估频率 50,000 步，训练在 49,999 步中断)

**保存的模型**:

| 文件名 | 步数 | 类型 |
|--------|------|------|
| `sac_bipedalwalker_final.zip` | 49,999 | 最终模型 (中断保存) |

**JSON 摘要路径**: `./logs/round_1/training_summary.json`

**备注**:
- 机器人约在 17,000 步后学会"半蹲不动"策略：通过保持低姿态、小幅度动作，让 episode 撑满 1600 步超时，避免摔倒的 -100 惩罚
- 原始奖励函数对"不动"过于宽容，导致智能体陷入局部最优
- 训练日志中 Step 级别的 Reward 显示为 0.00，是因为 logger 在非 episode 结束步读取不到 `episode` 字段

---

### 遇到的困难与解决方案

#### 困难 1: tqdm / rich 缺失导致训练无法启动

**现象**:
```
ImportError: You must install tqdm and rich in order to use the progress bar callback.
```

**原因**: `stable-baselines3` 的 `progress_bar=True` 依赖 `tqdm` 和 `rich`，但它们不是核心依赖。

**解决**:
```bash
pip install tqdm rich
```

**结果**: ✅ 解决，进度条正常显示。

---

#### 困难 2: 机器人"半蹲不动"局部最优 (核心问题)

**现象**:
- 训练到约 17,000 步后，大量 episode 达到最大长度 1600 步
- 评估窗口中机器人呈现半蹲姿态，几乎不移动
- 奖励稳定在 -100 左右，不再提升

**原因分析**:
- BipedalWalker 原始奖励：每步存活 `-0.00035`，摔倒 `-100`
- 机器人发现：**站着不动 1600 步 ≈ -0.56 总奖励** > **走路摔倒 ≈ -100 总奖励**
- 这是典型的**局部最优陷阱**，SAC 的熵机制也无法有效跳出

**解决**: 引入奖励塑形 (`reward_shaping.py`)

在 `train_sac.py` 中启用：
```python
USE_REWARD_SHAPING = True
FORWARD_WEIGHT = 2.0      # 奖励前进
STALL_PENALTY = 1.0       # 惩罚不动
ENABLE_EARLY_TERMINATION = True   # 卡住 100 步强制结束
```

**结果**: ✅ **已验证，效果显著！** 评估奖励从 -108 飙升至 +527，策略成功学会行走并跨越障碍。

---

#### 困难 3: 训练被中断后仅保存最终模型

**现象**: 按 Ctrl+C 后只保存了 `sac_bipedalwalker_final.zip`，没有中间检查点。

**原因**: 训练仅运行了 49,999 步，未达到第一个检查点保存阈值 (100,000 步)。

**解决**: 无需解决，属于正常行为。后续长训练会自动按 `CHECKPOINT_FREQ = 100_000` 保存中间模型。

---

---

### Round 2 — 预训练优化 (加载 Round 1 最终模型 + 奖励塑形)

| 项目 | 内容 |
|------|------|
| 轮次 | Round 2 |
| 训练步数 | **503,978** / 10,000,000 (约 5%) |
| 起始时间 | 2026-06-11 21:30:32 |
| 结束时间 | 2026-06-11 23:14:37 |
| 状态 | ⚠️ **用户中断** |
| RESUME_FROM | `./models/round_1/sac_bipedalwalker_final.zip` |
| 奖励塑形 | ✅ 启用 (`STALL_PENALTY=1.0`, `FORWARD_WEIGHT=2.0`) |
| 训练耗时 | 6,244.35 秒 (约 104 分钟 / 1.7 小时) |
| 平均 FPS | ~82 |

**训练统计**:
- 总 Episode 数: 数据未记录 (episodes 数组为空，可能因中断导致未写入)
- 评估奖励从 **-108.06** → **+527.78**，提升 **635+**

**评估记录（关键指标）**:

| 步数 | 平均奖励 | 标准差 | 阶段判断 |
|------|----------|--------|----------|
| 50,000 | -108.06 | 66.13 | 初期探索，仍受 Round 1 局部最优影响 |
| 100,000 | **+48.15** | 142.53 | 🎉 **首次转正！** 策略开始学会行走 |
| 150,000 | **+285.75** | 73.44 | 🚀 **大幅突破！** 成功跨越障碍 |
| 200,000 | +92.43 | 67.47 | 短暂回落，地形泛化不稳定 |
| 250,000 | +124.95 | 176.36 | 波动较大，遇到陌生地形 |
| 300,000 | +237.27 | 165.49 | 再次上升，经验积累中 |
| 350,000 | **+428.93** | 30.23 | ✅ **稳定高分**，std 大幅降低 |
| 400,000 | **+478.19** | 4.97 | ✅ **接近满分**，策略高度稳定 |
| 450,000 | **+522.35** | 1.42 | ✅ **几乎满分**，几乎每次都能跑完全程 |
| 500,000 | **+527.78** | 3.10 | ✅ **超过满分线**，完美行走 |

**保存的模型**:

| 文件名 | 步数 | 类型 |
|--------|------|------|
| `sac_bipedalwalker_100000_steps.zip` | 100,000 | 检查点 |
| `sac_bipedalwalker_200000_steps.zip` | 200,000 | 检查点 |
| `sac_bipedalwalker_300000_steps.zip` | 300,000 | 检查点 |
| `sac_bipedalwalker_400000_steps.zip` | 400,000 | 检查点 |
| `sac_bipedalwalker_500000_steps.zip` | 500,000 | 检查点 |
| `sac_bipedalwalker_final.zip` | 503,978 | 最终模型 (中断保存) |

**JSON 摘要路径**: `./logs/round_2/training_summary.json`

**备注**:
- 🎉 **奖励塑形验证成功！** 仅 50 万步就实现了从"躺平"到"满分行走"的跨越
- 100,000 步时奖励首次转正 (+48.15)，说明策略已跳出 Round 1 的局部最优
- 150,000 步时奖励飙升至 +285.75，这是**关键突破点**——机器人学会了有效跨障碍
- 350,000 步后标准差从 176 骤降至 30，说明策略**高度稳定**，不再随机摔倒
- 500,000 步时奖励 +527.78 超过 BipedalWalker 满分线 (~300+)，说明机器人能**稳定跑完全程**
- 训练被中断时策略已非常成熟，后续可加载 `500000_steps` 检查点继续训练或用于演示

---

### Round 3 — 奖励塑形强化 (加载 Round 2 的 50万步黄金检查点)

| 项目 | 内容 |
|------|------|
| 轮次 | Round 3 |
| 训练步数 | **643,052** / 10,000,000 (约 6.4%) |
| 起始时间 | 2026-06-11 23:42:09 |
| 结束时间 | 2026-06-12 01:37:31 |
| 状态 | ⚠️ **用户中断（策略退化，回退）** |
| RESUME_FROM | `./models/round_2/sac_bipedalwalker_500000_steps.zip` |
| 奖励塑形 | ✅ 启用 (增强版) |
| 超参数变更 | 见下方 |
| 训练耗时 | 6,922.26 秒 (约 115 分钟 / 1.9 小时) |
| 平均 FPS | ~92 |

**超参数变更**:

```python
# 网络结构
NET_ARCH = [256, 256]           # 保持与 Round 2 兼容

# 学习率
LEARNING_RATE = 1e-4            # ⚠️ 从 3e-4 降低，导致策略退化

# 探索
ENT_COEF = "auto"               # 保持自动调整

# 奖励塑形增强
FORWARD_WEIGHT = 5.0            # 从 2.0 → 5.0
STALL_PENALTY = 3.0             # 从 1.0 → 3.0
STALL_THRESHOLD = 0.15          # 从 0.05 → 0.15
```

**评估记录**:

| 步数 | 平均奖励 | 标准差 | 阶段判断 |
|------|----------|--------|----------|
| 50,000 | +14.43 | 9.37 | 加载 Round 2 后初期，策略适应新塑形 |
| 100,000 | +31.34 | 16.03 | 缓慢爬升，但远低于 Round 2 同期 |
| 200,000 | +32.56 | 5.93 | 停滞，学习率太低导致更新不足 |
| 300,000 | +26.20 | 11.18 | ⚠️ **下降**，策略开始不稳定 |
| 400,000 | +56.04 | 20.41 | 波动大，critic_loss 上升 |
| 500,000 | +47.59 | 32.77 | ⚠️ **远低于 Round 2 的 +527** |
| 550,000 | +294.38 | 145.37 | 突然飙升，但 std 极高，不稳定 |
| 600,000 | +419.74 | 8.35 | 有所恢复，但仍低于 Round 2 |

**训练统计（中断时）**:
- actor_loss: **-9.6** (⚠️ 为负，策略退化)
- critic_loss: 0.75~2.3 (波动上升)
- ent_coef: 0.0186 (探索收紧)

**保存的模型**:

| 文件名 | 步数 | 类型 |
|--------|------|------|
| `sac_bipedalwalker_100000_steps.zip` | 100,000 | 检查点 |
| `sac_bipedalwalker_200000_steps.zip` | 200,000 | 检查点 |
| `sac_bipedalwalker_300000_steps.zip` | 300,000 | 检查点 |
| `sac_bipedalwalker_400000_steps.zip` | 400,000 | 检查点 |
| `sac_bipedalwalker_500000_steps.zip` | 500,000 | 检查点 |
| `sac_bipedalwalker_600000_steps.zip` | 600,000 | 检查点 |
| `sac_bipedalwalker_final.zip` | 643,052 | 最终模型 (中断保存) |

**JSON 摘要路径**: `./logs/round_3/training_summary.json`

**备注**:
- ❌ **学习率 1e-4 过低 + 奖励塑形过强 = 策略退化**
- Round 3 的评估奖励最高仅 +419（60万步），远低于 Round 2 同期的 +527（50万步）
- actor_loss 为负 (-9.6) 是核心危险信号：Critic 被"骗"，认为当前策略很好，Actor 强化错误方向
- **结论**：`LEARNING_RATE = 1e-4` 无法有效适应增强后的奖励塑形，导致策略在"勉强在动"的局部最优中固化
- **决策**：回退到 Round 2 的 50万步黄金检查点，恢复 `LEARNING_RATE = 3e-4`，开始 Round 4

#### Round 3 遇到的困难与解决方案

**困难 1: `ENT_COEF` 改为固定值后模型加载失败**

**现象**:
```
AttributeError: 'NoneType' object has no attribute 'load_state_dict'
```

**原因**: Round 2 保存的模型使用 `ent_coef="auto"`（内部为可训练对象），Round 3 改为 `ent_coef=0.1`（float）后，`SAC.load()` 传参导致内部结构不匹配。

**解决尝试**: 修改 `load_or_create_model` 函数
- 加载时**不传 `ent_coef`** 参数，避免结构冲突
- 加载完成后**手动覆盖**为固定值 `0.1`
- 清除自动优化相关属性

**结果**: ⚠️ 加载成功，但训练时再次报错 `AttributeError: 'SAC' object has no attribute 'ent_coef_tensor'`

**最终解决**: **将 `ENT_COEF` 改回 `"auto"`**
- SB3 的 `SAC` 在 `ent_coef="auto"` 和固定值之间切换时，内部状态机不兼容
- 加载 `"auto"` 模型后，无法安全地改为固定值
- Round 2 已用 `"auto"` 取得 +527.78 的成绩，说明自动调整在此任务上有效
- Round 3 的核心改进应聚焦于**奖励塑形**而非探索机制

**代码变更** (`train_sac.py`):
```python
# 改回自动调整
ENT_COEF = "auto"  # 自动调整温度系数 alpha

# load_or_create_model 中添加保护逻辑
if ENT_COEF != "auto" and hasattr(model, 'ent_coef'):
    if not hasattr(model, 'log_ent_coef'):
        # 仅当原始模型也是固定值时才允许覆盖
        model.ent_coef = th.tensor([float(ENT_COEF)], device=model.device)
    else:
        print(f"[警告] 原始模型使用 ent_coef='auto'，忽略配置中的固定值，保持自动调整")
```

**结果**: ✅ 解决，Round 3 成功加载并正常训练。

---

**困难 2: `LEARNING_RATE = 1e-4` 导致策略退化**

**现象**:
- 训练 64 万步后评估奖励仅 +419，远低于 Round 2 的 +527
- actor_loss 为负 (-9.6)，critic_loss 波动上升
- 策略在"勉强在动"的状态中固化，无法进化到更优的大步行走

**原因分析**:
- `LEARNING_RATE = 1e-4` 太低，策略网络更新太慢
- 同时 `STALL_PENALTY = 3.0` + `FORWARD_WEIGHT = 5.0` + `STALL_THRESHOLD = 0.15` 塑形过强
- 低学习率无法有效适应新的奖励 landscape，导致 Critic 对"勉强在动"的动作给出虚高 Q 值
- Actor 强化了这个错误策略，形成恶性循环

**解决**: **回退到 Round 2 配置，开始 Round 4**
- 加载 Round 2 的 50万步黄金检查点 (+527.78)
- 恢复 `LEARNING_RATE = 3e-4`
- 保留增强的奖励塑形参数，验证是否是学习率单一因素导致的问题
- 如果仍退化，再降低塑形强度

**结果**: ⏳ Round 4 待验证

---

### Round 4 — 回退优化 (加载 Round 2 黄金检查点，恢复学习率)

| 项目 | 内容 |
|------|------|
| 轮次 | Round 4 |
| 训练步数 | ⏳ 待定 |
| 起始时间 | ⏳ 待定 |
| 结束时间 | ⏳ 待定 |
| 状态 | ⏳ 未开始 |
| RESUME_FROM | `./models/round_2/sac_bipedalwalker_500000_steps.zip` |
| 奖励塑形 | ✅ 启用 (Round 3 增强版) |
| 超参数变更 | 见下方 |

**超参数变更**:

```python
# 网络结构
NET_ARCH = [256, 256]           # 保持与 Round 2 兼容

# 学习率
LEARNING_RATE = 3e-4            # ✅ 从 1e-4 恢复，验证学习率是否为退化主因

# 探索
ENT_COEF = "auto"               # 保持自动调整

# 奖励塑形（保留 Round 3 增强版）
FORWARD_WEIGHT = 5.0            # 从 2.0 → 5.0
STALL_PENALTY = 3.0             # 从 1.0 → 3.0
STALL_THRESHOLD = 0.15          # 从 0.05 → 0.15
```

**核心思路**:
- 回退到 Round 2 的 50万步黄金检查点（+527.78）
- 只改一个变量：`LEARNING_RATE` 从 1e-4 → 3e-4
- 保留 Round 3 的增强塑形参数，验证是否是学习率单一因素导致退化
- 如果 10 万步后评估奖励回升到 +520+，说明学习率是关键
- 如果仍退化，再降低塑形强度（`STALL_PENALTY = 2.0`）

**预期效果**:
- 评估奖励目标：回到 +520~530 并稳定
- 消除小碎步，实现更连贯的大步行走

**JSON 摘要路径**: `./logs/round_4/training_summary.json`

**备注**:
- Round 4 是**控制变量实验**：只改学习率，其他保持不变
- 这是验证"1e-4 是否导致退化"的最直接方法
- 如果成功，说明 Round 3 的塑形方向是对的，只是学习率太低

---

## 多轮对比总结

| 轮次 | 起始方式 | 总步数 | 最高奖励 | 最终奖励 | 训练耗时 | 备注 |
|------|----------|--------|----------|----------|----------|------|
| 1 | 从头 | 49,999 | -48.61 | -109.88 (均值) | 499s | 基线，陷入局部最优 |
| 2 | 预训练 + 奖励塑形 | 503,978 | **+527.78** | **+527.78** | 6244s | 🎉 **成功学会行走** |
| 3 | Round 2 50万步 + 增强塑形 | 643,052 | +419.74 | +419.74 | 6922s | ⚠️ **策略退化**，学习率 1e-4 过低 |
| 4 | Round 2 50万步 + 恢复学习率 | ⏳ | ⏳ | ⏳ | ⏳ | 控制变量实验，验证学习率影响 |

---

## 训练难点备忘

1. **超参数敏感**: 网络层数、Batch Size、经验回放池容量设置不合理时，机器人会在原地疯狂前空翻或直接趴下
2. **训练时间长**: 通常需要 500万~1000万步以上才能看到机器人跌跌撞撞跨越障碍
3. **地形随机**: 每次 episode 的台阶、深坑、木桩位置随机生成，要求策略具有强泛化能力
4. **SAC 优势**: 最大熵机制和自适应温度系数 α 使其在新地形上比 DDPG 更不容易卡死
5. **渲染开销**: `render=True` 弹出 pygame 窗口时会暂停训练，仅用于观察，正式跑数据建议关闭
6. **局部最优陷阱**: 机器人容易学会"半蹲不动"——因为原始奖励中摔倒惩罚 (-100) 远大于存活惩罚 (-0.00035/步)。必须通过奖励塑形 (`USE_REWARD_SHAPING=True`) 或增大探索来打破

---

## 常用命令

```bash
# 启动训练 (当前配置)
python train_sac.py

# 查看 TensorBoard (所有轮次)
tensorboard --logdir=./logs

# 查看 TensorBoard (单轮)
tensorboard --logdir=./logs/round_1

# 测试环境
python test_env.py

# 测试奖励塑形 (500步随机动作，观察原始奖励 vs 塑形奖励)
python reward_shaping.py

# 查看 JSON 摘要
cat ./logs/round_1/training_summary.json

# 批量对比多轮结果 (Python)
python -c "
import json, glob
for path in sorted(glob.glob('./logs/round_*/training_summary.json')):
    with open(path) as f:
        d = json.load(f)
    r = d['training']['stats']
    print(f\"{path}: mean={r['mean_reward']}, max={r['max_reward']}, episodes={r['total_episodes']}\")
"
```

---

## 待办事项

- [ ] **Round 2**: 完成 1000 万步基线训练（已启用奖励塑形）
- [ ] **调参实验**: 如果 Round 1 仍卡住，尝试增大 `STALL_PENALTY` 到 2.0 或 3.0
- [ ] **Round 3**: 尝试降低学习率 (1e-4) 并增强奖励塑形
- [ ] **对比分析**: 整理多轮 JSON 摘要，绘制奖励曲线对比图
- [ ] **模型测试**: 加载最佳模型，录制一段机器人行走视频

---

## 参考

- [Gymnasium BipedalWalker](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [README.md](./README.md) — 项目完整说明文档
