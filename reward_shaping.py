"""
BipedalWalker-v3 奖励塑形包装器
解决机器人"躺平/半蹲不动"的局部最优问题

使用方式:
    from reward_shaping import make_shaped_env
    env = make_shaped_env(hardcore=True)
"""

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper


class BipedalWalkerRewardShaping(Wrapper):
    """
    对 BipedalWalker-v3 的奖励进行塑形，鼓励前进、惩罚不动。
    
    设计思路:
    1. 前进速度奖励 —— 与 torso 的水平速度成正比，走得越快奖励越高
    2. 站立姿态奖励 —— 躯干角度越接近垂直，奖励越高
    3. 不动惩罚 —— 当水平速度接近 0 时，给予额外负奖励
    4. 关节抖动惩罚 —— 动作变化过大时惩罚，鼓励平滑行走
    """

    def __init__(self, env, forward_weight=2.0, upright_weight=0.5,
                 stall_penalty=1.0, smooth_weight=0.1):
        """
        参数:
            forward_weight: 前进速度奖励权重 (默认 2.0)
            upright_weight: 站立姿态奖励权重 (默认 0.5)
            stall_penalty: 不动惩罚强度 (默认 1.0)
            smooth_weight: 动作平滑奖励权重 (默认 0.1)
        """
        super().__init__(env)
        self.forward_weight = forward_weight
        self.upright_weight = upright_weight
        self.stall_penalty = stall_penalty
        self.smooth_weight = smooth_weight
        
        # 记录上一帧动作，用于计算动作变化
        self.last_action = None
        
        # 记录原始奖励统计
        self.raw_reward_sum = 0.0
        self.shaped_reward_sum = 0.0
        self.step_count = 0

    def reset(self, **kwargs):
        self.last_action = None
        self.raw_reward_sum = 0.0
        self.shaped_reward_sum = 0.0
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 保存原始奖励
        raw_reward = reward
        
        # ========== 奖励塑形 ==========
        shaped = 0.0
        
        # 1. 前进速度奖励 (从 obs 中提取 torso 水平速度)
        # obs[0] = hull angle speed (角速度)
        # obs[1] = hull x velocity (水平速度，前进为正)
        # obs[2] = hull y velocity (垂直速度)
        x_velocity = obs[1] if len(obs) > 1 else 0.0
        
        # 只有当速度为正（前进）时才给奖励，后退不给
        if x_velocity > 0:
            shaped += self.forward_weight * x_velocity
        else:
            # 后退或静止时给予轻微惩罚
            shaped += self.forward_weight * x_velocity * 0.5
        
        # 2. 站立姿态奖励 (obs[3] = hull angle，0 表示垂直)
        # 角度越接近 0（垂直），奖励越高
        hull_angle = obs[3] if len(obs) > 3 else 0.0
        upright_bonus = self.upright_weight * (1.0 - abs(hull_angle))
        shaped += upright_bonus
        
        # 3. 不动惩罚 —— 关键！打破"站着不动"的局部最优
        # 当水平速度绝对值 < 0.1 时，认为机器人在"卡住"
        if abs(x_velocity) < 0.1:
            shaped -= self.stall_penalty
        
        # 4. 动作平滑奖励 (鼓励连贯动作，减少抖动)
        if self.last_action is not None:
            action_diff = np.mean(np.abs(action - self.last_action))
            shaped -= self.smooth_weight * action_diff
        
        self.last_action = np.array(action)
        
        # 合并奖励
        total_reward = raw_reward + shaped
        
        # 统计
        self.raw_reward_sum += raw_reward
        self.shaped_reward_sum += shaped
        self.step_count += 1
        
        # 把塑形信息放入 info，方便调试
        info["raw_reward"] = raw_reward
        info["shaped_reward"] = shaped
        info["x_velocity"] = x_velocity
        info["upright_bonus"] = upright_bonus
        
        # 每 100 步在 info 中打印一次累计统计（用于观察）
        if self.step_count % 100 == 0:
            info["reward_stats"] = {
                "steps": self.step_count,
                "raw_sum": round(self.raw_reward_sum, 2),
                "shaped_sum": round(self.shaped_reward_sum, 2),
            }
        
        return obs, total_reward, terminated, truncated, info


class EarlyTerminationWrapper(Wrapper):
    """
    提前终止 Wrapper：当机器人明显卡住不动超过一定步数时，提前结束 episode。
    避免浪费训练时间在"躺平"状态上。
    """
    
    def __init__(self, env, stall_threshold=0.05, max_stall_steps=100):
        super().__init__(env)
        self.stall_threshold = stall_threshold
        self.max_stall_steps = max_stall_steps
        self.stall_counter = 0
    
    def reset(self, **kwargs):
        self.stall_counter = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 检测是否卡住
        x_velocity = obs[1] if len(obs) > 1 else 0.0
        if abs(x_velocity) < self.stall_threshold:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
        
        # 如果连续卡住超过阈值，强制终止
        if self.stall_counter >= self.max_stall_steps:
            terminated = True
            info["early_termination"] = "stall"
            info["stall_steps"] = self.stall_counter
        
        return obs, reward, terminated, truncated, info


def make_shaped_env(hardcore=True, render_mode=None, 
                    forward_weight=2.0, upright_weight=0.5,
                    stall_penalty=1.0, smooth_weight=0.1,
                    enable_early_termination=True,
                    stall_threshold=0.05, max_stall_steps=100):
    """
    创建带奖励塑形的 BipedalWalker 环境。
    
    参数:
        hardcore: 是否困难模式
        render_mode: 渲染模式
        forward_weight: 前进奖励权重
        upright_weight: 站立奖励权重  
        stall_penalty: 不动惩罚强度
        smooth_weight: 动作平滑权重
        enable_early_termination: 是否启用提前终止
        stall_threshold: 判定为卡住的水平速度阈值
        max_stall_steps: 允许连续卡住的最大步数
    
    返回:
        包装后的环境
    """
    env = gym.make("BipedalWalker-v3", hardcore=hardcore, render_mode=render_mode)
    
    # 先加奖励塑形
    env = BipedalWalkerRewardShaping(
        env,
        forward_weight=forward_weight,
        upright_weight=upright_weight,
        stall_penalty=stall_penalty,
        smooth_weight=smooth_weight,
    )
    
    # 再加提前终止（可选）
    if enable_early_termination:
        env = EarlyTerminationWrapper(env, stall_threshold, max_stall_steps)
    
    return env


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("奖励塑形环境测试")
    print("=" * 50)
    
    env = make_shaped_env(hardcore=True, render_mode=None)
    obs, info = env.reset(seed=42)
    
    print(f"观测空间: {obs.shape}")
    print(f"动作空间: {env.action_space.shape}")
    
    total_reward = 0.0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 100 == 0:
            print(f"Step {step:4d}: reward={reward:7.3f}  "
                  f"raw={info.get('raw_reward', 0):7.3f}  "
                  f"shaped={info.get('shaped_reward', 0):7.3f}  "
                  f"x_vel={info.get('x_velocity', 0):6.3f}")
        
        if terminated or truncated:
            print(f"Episode 结束于 step {step}, 总奖励: {total_reward:.2f}")
            if "early_termination" in info:
                print(f"  原因: 提前终止 ({info['early_termination']})")
            break
    
    env.close()
    print("=" * 50)
    print("测试完成！")
    print("=" * 50)
