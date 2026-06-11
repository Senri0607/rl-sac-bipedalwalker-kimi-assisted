"""
环境测试脚本
用于验证 BipedalWalker-v3 Hardcore 环境是否能正常创建和运行
"""

import gymnasium as gym
import numpy as np

ENV_NAME = "BipedalWalker-v3"
HARDCORE = True


def test_env():
    print("=" * 50)
    print("BipedalWalker-v3 Hardcore 环境测试")
    print("=" * 50)

    try:
        env = gym.make(ENV_NAME, hardcore=HARDCORE, render_mode=None)
        print(f"[OK] 环境创建成功: {ENV_NAME} (hardcore={HARDCORE})")
    except Exception as e:
        print(f"[FAIL] 环境创建失败: {e}")
        return False

    try:
        obs, info = env.reset(seed=42)
        print(f"[OK] 环境重置成功")
        print(f"  观测空间形状: {obs.shape}")
        print(f"  观测空间范围: [{env.observation_space.low.min():.2f}, {env.observation_space.high.max():.2f}]")
        print(f"  动作空间形状: {env.action_space.shape}")
        print(f"  动作空间范围: [{env.action_space.low.min():.2f}, {env.action_space.high.max():.2f}]")
    except Exception as e:
        print(f"[FAIL] 环境重置失败: {e}")
        return False

    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"[OK] 单步执行成功")
        print(f"  奖励: {reward:.4f}")
        print(f"  是否结束: {terminated or truncated}")
    except Exception as e:
        print(f"[FAIL] 单步执行失败: {e}")
        return False

    try:
        env.close()
        print(f"[OK] 环境关闭成功")
    except Exception as e:
        print(f"[FAIL] 环境关闭失败: {e}")
        return False

    print("=" * 50)
    print("[PASS] 环境测试全部通过！")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_env()
    exit(0 if success else 1)
