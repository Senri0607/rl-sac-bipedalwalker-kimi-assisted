"""
SAC (Soft Actor-Critic) 训练脚本
针对 BipedalWalker-v3 Hardcore 环境
默认使用 GPU 训练

支持三种多轮训练模式：
1. 完全从头 (ROUND_ID 递增, RESUME_FROM=None)
2. 加载预训练当起点 (ROUND_ID 递增, RESUME_FROM=上一轮的模型路径)
3. 继续训练 (ROUND_ID 不变, RESUME_FROM=当前轮的检查点)
"""

import json
import os
import sys
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from reward_shaping import make_shaped_env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ==================== 配置区域 ====================
ENV_NAME = "BipedalWalker-v3"
HARDCORE = True
RENDER_MODE = None  # 训练时不渲染，设为 "rgb_array" 可录制

# 训练参数
TOTAL_TIMESTEPS = 10_000_000  # 本轮要训练的总步数
LEARNING_RATE = 3e-4          # Round 6: 保持成功学习率
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
TAU = 0.005
GAMMA = 0.99
ENT_COEF = "auto"  # 自动调整温度系数 alpha
# 注意: 加载已保存模型时，不要改为固定值，会导致 SB3 内部属性冲突

# 网络结构
NET_ARCH = [256, 256]         # 保持与 Round 5 兼容，加载已有模型

# 日志与保存
ROUND_ID = 6                  # 轮次编号，每轮递增，自动创建独立目录
CHECKPOINT_FREQ = 100_000
EVAL_FREQ = 50_000
EVAL_EPISODES = 2

# 多轮训练控制 (三选一)
# -------------------------------------------------
# 模式 A: 完全从头 —— 保持 None
# 模式 B: 加载预训练当新轮次起点 —— 填上一轮模型路径
# 模式 C: 继续训练 —— 填当前轮检查点路径
# Round 6: 从 Round 5 的 45万步检查点 (+572.51) 继续，温和增加塑形
RESUME_FROM = "./models/round_5/sac_bipedalwalker_final.zip"
# -------------------------------------------------

# 奖励塑形 —— 解决机器人"卡住不动"的局部最优问题
# 启用后会额外奖励前进、惩罚不动，强烈建议困难版开启
USE_REWARD_SHAPING = True

# 奖励塑形参数 (仅在 USE_REWARD_SHAPING=True 时生效)
# Round 6: 温和增强塑形，修正 obs 维度，增加步态奖励
FORWARD_WEIGHT = 2.5        # ✅ 从 2.0 → 2.5，温和增加大步前进奖励
UPRIGHT_WEIGHT = 0.5        # 站立姿态奖励权重
STALL_PENALTY = 1.0         # 保持 Round 5 成功值
SMOOTH_WEIGHT = 0.1         # 动作平滑权重
LIFT_WEIGHT = 0.1           # ⭐ 新增：抬腿跨步奖励权重，鼓励跨越障碍
STRIDE_WEIGHT = 0.05        # ⭐ 新增：空中步态奖励权重，鼓励正常交替步态
ENABLE_EARLY_TERMINATION = True   # 卡住时是否提前终止 episode
STALL_THRESHOLD = 0.05      # 保持 Round 5 成功值
MAX_STALL_STEPS = 100       # 允许连续卡住的最大步数

# 注意：render=True 会弹出 pygame 窗口显示机器人走路，但会拖慢训练速度。
# 如果不需要观看，把下面 eval_env 的 render_mode 改回 None，render 改回 False。

# 设备配置：默认使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 路径构建 ====================

LOG_DIR = f"./logs/round_{ROUND_ID}"
MODEL_DIR = f"./models/round_{ROUND_ID}"


def make_env(hardcore=True, render_mode=None):
    """创建并包装环境（支持奖励塑形）"""
    def _init():
        if USE_REWARD_SHAPING:
            env = make_shaped_env(
                hardcore=hardcore,
                render_mode=render_mode,
                forward_weight=FORWARD_WEIGHT,
                upright_weight=UPRIGHT_WEIGHT,
                stall_penalty=STALL_PENALTY,
                smooth_weight=SMOOTH_WEIGHT,
                lift_weight=LIFT_WEIGHT,
                stride_weight=STRIDE_WEIGHT,
                enable_early_termination=ENABLE_EARLY_TERMINATION,
                stall_threshold=STALL_THRESHOLD,
                max_stall_steps=MAX_STALL_STEPS,
            )
        else:
            env = gym.make(
                ENV_NAME,
                hardcore=hardcore,
                render_mode=render_mode,
            )
            env = Monitor(env)
        return env
    return _init


class TrainingLoggerCallback(BaseCallback):
    """自定义回调：记录训练指标到文件"""

    def __init__(self, log_file="training_log.txt", verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        header = (
            f"{'Step':>10} | {'Time':>8} | {'Reward':>10} | "
            f"{'Episode Len':>12} | {'FPS':>6} | {'Loss':>10}\n"
        )
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备: {device}\n")
            f.write(f"环境: {ENV_NAME} (hardcore={HARDCORE})\n")
            f.write(f"算法: SAC\n")
            f.write(f"总步数: {TOTAL_TIMESTEPS:,}\n")
            f.write("=" * 70 + "\n")
            f.write(header)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            elapsed = time.time() - self.start_time
            fps = int(self.n_calls / elapsed) if elapsed > 0 else 0
            info = self.locals.get("infos", [{}])[0]
            reward = info.get("episode", {}).get("r", 0)
            length = info.get("episode", {}).get("l", 0)

            line = (
                f"{self.n_calls:>10} | {elapsed:>7.0f}s | {reward:>10.2f} | "
                f"{length:>12} | {fps:>6} | {'N/A':>10}\n"
            )
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line)
        return True


class JsonSummaryCallback(BaseCallback):
    """
    收集训练全过程数据，训练结束时导出 JSON 摘要。
    支持多轮训练：每轮独立记录，可对比。
    """

    def __init__(self, log_dir="./logs", round_id=1, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.round_id = round_id
        self.start_time = None
        self.data = {
            "project": "BipedalWalker-v3 Hardcore SAC",
            "version": "1.0",
            "round_id": round_id,
            "config": {},
            "system": {},
            "training": {
                "start_time": None,
                "end_time": None,
                "total_timesteps": 0,
                "elapsed_seconds": 0.0,
                "episodes": [],
            },
            "evaluations": [],
            "checkpoints": [],
            "final_model": None,
            "status": "running",
            "notes": [],
        }

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.data["training"]["start_time"] = datetime.now().isoformat()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.data["training"]["episodes"].append({
                    "step": int(self.num_timesteps),
                    "reward": float(ep.get("r", 0.0)),
                    "length": int(ep.get("l", 0)),
                    "elapsed_seconds": round(time.time() - self.start_time, 2),
                })
        return True

    def _on_training_end(self) -> None:
        self._save()

    def _save(self):
        self.data["training"]["end_time"] = datetime.now().isoformat()
        self.data["training"]["total_timesteps"] = int(self.num_timesteps)
        if self.start_time:
            self.data["training"]["elapsed_seconds"] = round(
                time.time() - self.start_time, 2
            )

        episodes = self.data["training"]["episodes"]
        if episodes:
            rewards = [ep["reward"] for ep in episodes]
            lengths = [ep["length"] for ep in episodes]
            self.data["training"]["stats"] = {
                "total_episodes": len(episodes),
                "mean_reward": round(float(np.mean(rewards)), 4),
                "max_reward": round(float(np.max(rewards)), 4),
                "min_reward": round(float(np.min(rewards)), 4),
                "std_reward": round(float(np.std(rewards)), 4),
                "mean_length": round(float(np.mean(lengths)), 2),
            }

        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "training_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"\n[JSON 摘要] 已保存: {path}")

    def record_config(self, **kwargs):
        self.data["config"].update(kwargs)

    def record_system(self, **kwargs):
        self.data["system"].update(kwargs)

    def record_eval(self, step, mean_reward, std_reward):
        self.data["evaluations"].append({
            "step": int(step),
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "elapsed_seconds": round(time.time() - self.start_time, 2) if self.start_time else 0.0,
        })

    def record_checkpoint(self, step, path):
        self.data["checkpoints"].append({
            "step": int(step),
            "path": str(path),
            "elapsed_seconds": round(time.time() - self.start_time, 2) if self.start_time else 0.0,
        })

    def set_final_model(self, path):
        self.data["final_model"] = str(path)

    def set_status(self, status):
        self.data["status"] = status

    def add_note(self, note):
        self.data["notes"].append({
            "time": datetime.now().isoformat(),
            "content": str(note),
        })


def load_or_create_model(train_env, resume_path=None):
    """
    根据 resume_path 决定是加载已有模型还是创建新模型。
    
    参数:
        train_env: 训练环境
        resume_path: 模型文件路径 (.zip)
    
    返回:
        model: SAC 模型实例
        is_resumed: 是否从已有模型加载
    """
    if resume_path and os.path.exists(resume_path):
        print(f"\n[加载模型] 从 {resume_path} 加载...")
        # 加载时不传 ent_coef，避免与保存模型的结构冲突
        model = SAC.load(
            resume_path,
            env=train_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            tensorboard_log=LOG_DIR,
            verbose=1,
            device=device,
        )
        # 加载后手动覆盖 ent_coef（如果配置为固定值）
        # 注意: 由于 SB3 内部机制限制，加载 "auto" 模型后不建议改为固定值
        # 如需固定 ent_coef，请创建新模型而非加载旧模型
        if ENT_COEF != "auto" and hasattr(model, 'ent_coef'):
            import torch as th
            # 仅当原始模型也是固定值时才允许覆盖
            if not hasattr(model, 'log_ent_coef'):
                model.ent_coef = th.tensor([float(ENT_COEF)], device=model.device)
                print(f"[加载模型] ent_coef 已手动设置为 {ENT_COEF}")
            else:
                print(f"[警告] 原始模型使用 ent_coef='auto'，忽略配置中的固定值，保持自动调整")
        print("[加载模型] 成功！")
        print("[加载模型] 成功！")
        return model, True
    else:
        if resume_path:
            print(f"\n[警告] 指定模型不存在: {resume_path}")
            print("[警告] 将创建全新模型...")
        print("\n[创建模型] 初始化全新 SAC 模型...")
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            ent_coef=ENT_COEF,
            policy_kwargs=dict(net_arch=NET_ARCH),
            tensorboard_log=LOG_DIR,
            verbose=1,
            device=device,
        )
        return model, False


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("BipedalWalker-v3 Hardcore - SAC 训练")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"使用设备: {device}")
    print(f"环境: {ENV_NAME} (hardcore={HARDCORE})")
    print(f"本轮次: Round {ROUND_ID}")
    print(f"本轮训练步数: {TOTAL_TIMESTEPS:,}")
    if RESUME_FROM:
        print(f"加载来源: {RESUME_FROM}")
    if USE_REWARD_SHAPING:
        print(f"奖励塑形: 已启用 (forward={FORWARD_WEIGHT}, stall_penalty={STALL_PENALTY})")
    else:
        print("奖励塑形: 未启用")
    print("=" * 60)

    # 创建目录
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 创建环境
    print("\n[1/4] 创建训练环境...")
    train_env = DummyVecEnv([make_env(hardcore=HARDCORE, render_mode=RENDER_MODE)])

    # 创建评估环境（启用可视化渲染）
    eval_env = DummyVecEnv([make_env(hardcore=HARDCORE, render_mode="human")])

    # 创建或加载 SAC 模型
    print("[2/4] 初始化 SAC 模型...")
    model, is_resumed = load_or_create_model(train_env, RESUME_FROM)

    # 回调函数
    print("[3/4] 配置回调函数...")

    json_callback = JsonSummaryCallback(log_dir=LOG_DIR, round_id=ROUND_ID)
    json_callback.record_config(
        env_name=ENV_NAME,
        hardcore=HARDCORE,
        algorithm="SAC",
        total_timesteps=TOTAL_TIMESTEPS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        tau=TAU,
        gamma=GAMMA,
        ent_coef=str(ENT_COEF),
        net_arch=NET_ARCH,
        eval_freq=EVAL_FREQ,
        eval_episodes=EVAL_EPISODES,
        checkpoint_freq=CHECKPOINT_FREQ,
        resume_from=RESUME_FROM,
        is_resumed=is_resumed,
        use_reward_shaping=USE_REWARD_SHAPING,
        forward_weight=FORWARD_WEIGHT,
        upright_weight=UPRIGHT_WEIGHT,
        stall_penalty=STALL_PENALTY,
        smooth_weight=SMOOTH_WEIGHT,
        enable_early_termination=ENABLE_EARLY_TERMINATION,
        stall_threshold=STALL_THRESHOLD,
        max_stall_steps=MAX_STALL_STEPS,
    )
    json_callback.record_system(
        pytorch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        device=device,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_DIR,
        name_prefix="sac_bipedalwalker",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=True,
        n_eval_episodes=EVAL_EPISODES,
    )

    logger_callback = TrainingLoggerCallback(
        log_file=os.path.join(LOG_DIR, "training_log.txt")
    )

    # 开始训练
    print("[4/4] 开始训练！")
    print(f"TensorBoard 日志: {LOG_DIR}")
    print(f"模型保存目录: {MODEL_DIR}")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback, logger_callback, json_callback],
            progress_bar=True,
        )
        json_callback.set_status("completed")
        json_callback.add_note("训练正常完成，达到目标步数。")
    except KeyboardInterrupt:
        print("\n训练被用户中断。")
        json_callback.set_status("interrupted")
        json_callback.add_note("训练被用户手动中断 (Ctrl+C)。")
    finally:
        # 保存最终模型
        final_path = os.path.join(MODEL_DIR, "sac_bipedalwalker_final.zip")
        model.save(final_path)
        json_callback.set_final_model(final_path)
        print(f"\n最终模型已保存: {final_path}")

        # 尝试读取 EvalCallback 保存的评估数据
        eval_npz_path = os.path.join(LOG_DIR, "evaluations.npz")
        if os.path.exists(eval_npz_path):
            try:
                eval_data = np.load(eval_npz_path)
                timesteps = eval_data.get("timesteps", [])
                results = eval_data.get("results", [])
                for i, (step, result) in enumerate(zip(timesteps, results)):
                    json_callback.record_eval(
                        step=int(step),
                        mean_reward=float(np.mean(result)),
                        std_reward=float(np.std(result)),
                    )
                json_callback.add_note(f"从 evaluations.npz 导入了 {len(timesteps)} 次评估记录。")
            except Exception as e:
                json_callback.add_note(f"读取 evaluations.npz 失败: {e}")

        # 扫描已保存的检查点
        if os.path.exists(MODEL_DIR):
            for fname in sorted(os.listdir(MODEL_DIR)):
                if fname.startswith("sac_bipedalwalker_") and fname.endswith(".zip"):
                    try:
                        step_str = fname.replace("sac_bipedalwalker_", "").replace("_steps.zip", "").replace("_best_model.zip", "").replace("_final.zip", "")
                        step = int(step_str) if step_str.isdigit() else 0
                    except ValueError:
                        step = 0
                    json_callback.record_checkpoint(
                        step=step,
                        path=os.path.join(MODEL_DIR, fname),
                    )

        # 确保 JSON 摘要被保存
        json_callback._save()

        # 关闭环境
        train_env.close()
        eval_env.close()

    print("\n训练完成！")
    print(f"查看 TensorBoard: tensorboard --logdir={LOG_DIR}")
    print(f"查看 JSON 摘要: {os.path.join(LOG_DIR, 'training_summary.json')}")
    print(f"\n下一轮建议:")
    print(f"  从头开始新轮次: 改 ROUND_ID = {ROUND_ID + 1}, RESUME_FROM = None")
    print(f"  用本轮结果当起点: 改 ROUND_ID = {ROUND_ID + 1}, RESUME_FROM = \"{final_path}\"")
    print(f"  继续本轮训练: 保持 ROUND_ID = {ROUND_ID}, RESUME_FROM = \"./models/round_{ROUND_ID}/sac_bipedalwalker_xxx_steps.zip\"")


if __name__ == "__main__":
    main()
