"""
BipedalWalker-v3 Hardcore 模型评估程序
功能：
1. 选择模型路径，默认100次评估
2. 使用matplotlib量化数据并保存
3. 简易可视化窗口（tkinter）
4. 评估过程录屏并保存
5. 实时pygame显示评估动画

使用方法：
    python eval_model.py
"""

import os
import sys
import threading
import time
from datetime import datetime
from tkinter import Tk, Label, Button, Entry, Checkbutton, BooleanVar, StringVar, filedialog, messagebox, scrolledtext, Frame

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reward_shaping import make_shaped_env
from stable_baselines3 import SAC

matplotlib.use("Agg")  # 无头模式，不弹窗

# ------------------- 配置 -------------------
DEFAULT_EVAL_EPISODES = 100
VIDEO_FPS = 30

# 确保保存目录存在
os.makedirs("./eval_results", exist_ok=True)


# ------------------- pygame 实时显示窗口 -------------------

class PygameViewer:
    """用于在评估过程中实时显示 pygame 动画窗口"""
    
    def __init__(self, width=600, height=400, title="BipedalWalker Eval"):
        self.width = width
        self.height = height
        self.title = title
        self.screen = None
        self.clock = None
        self._running = False
        self._init_pygame()
    
    def _init_pygame(self):
        try:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.title)
            self.clock = pygame.time.Clock()
            self._running = True
        except ImportError:
            print("[警告] pygame 未安装，无法显示实时窗口。请运行: pip install pygame")
            self._running = False
    
    def show_frame(self, frame_rgb):
        """显示一帧 RGB 图像到 pygame 窗口"""
        if not self._running or self.screen is None:
            return
        
        import pygame
        
        # 处理 pygame 事件（防止窗口卡死）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return
        
        # 确保帧是 uint8 格式
        if frame_rgb.dtype != np.uint8:
            frame_rgb = (frame_rgb * 255).astype(np.uint8) if frame_rgb.max() <= 1.0 else frame_rgb.astype(np.uint8)
        
        # 调整大小适应窗口
        if frame_rgb.shape[0] != self.height or frame_rgb.shape[1] != self.width:
            frame_rgb = np.array(Image.fromarray(frame_rgb).resize((self.width, self.height)))
        
        # 转置为 pygame 格式 (width, height, 3) -> (width, height)
        surface = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        
        # 控制帧率，避免CPU占用过高
        self.clock.tick(VIDEO_FPS)
    
    def close(self):
        """关闭 pygame 窗口"""
        if self._running and self.screen is not None:
            import pygame
            pygame.quit()
            self._running = False


# ------------------- 评估核心逻辑 -------------------

def evaluate_model(model_path, num_episodes, use_shaping, render_video, video_path, log_callback, show_pygame=True):
    """
    评估模型核心逻辑
    
    参数:
        model_path: 模型文件路径
        num_episodes: 评估次数
        use_shaping: 是否使用奖励塑形
        render_video: 是否录屏
        video_path: 视频保存路径
        log_callback: 回调函数(str) -> 更新GUI日志
        show_pygame: 是否显示 pygame 实时窗口
    
    返回:
        results: 字典，包含 rewards, lengths, frames 等
    """
    
    # 创建 pygame 窗口（如果需要实时显示）
    pygame_window = None
    if show_pygame:
        try:
            pygame_window = PygameViewer(width=600, height=400, title=f"Eval: {os.path.basename(model_path)}")
            if pygame_window._running:
                log_callback("[✓] pygame 实时窗口已创建")
            else:
                log_callback("[✗] pygame 窗口创建失败，继续无窗口评估")
                pygame_window = None
        except Exception as e:
            log_callback(f"[✗] pygame 初始化失败: {e}")
            pygame_window = None
    
    # 创建环境 —— 始终使用 rgb_array 以便获取帧数据
    # 然后手动用 pygame 显示
    render_mode = "rgb_array" if (render_video or show_pygame) else None
    
    if use_shaping:
        env = make_shaped_env(
            hardcore=True,
            render_mode=render_mode,
            forward_weight=2.5,
            upright_weight=0.5,
            stall_penalty=1.0,
            smooth_weight=0.1,
            lift_weight=0.1,
            stride_weight=0.05,
            enable_early_termination=True,
            stall_threshold=0.05,
            max_stall_steps=100,
        )
    else:
        env = gym.make("BipedalWalker-v3", hardcore=True, render_mode=render_mode)
    
    # 加载模型
    try:
        model = SAC.load(model_path, env=env)
        log_callback(f"[✓] 模型加载成功: {model_path}")
    except Exception as e:
        log_callback(f"[✗] 模型加载失败: {e}")
        env.close()
        if pygame_window:
            pygame_window.close()
        return None
    
    rewards = []
    lengths = []
    all_frames = []  # 用于录屏
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_frames = []
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # 获取帧（rgb_array 模式）
            if render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    # 保存到视频
                    if render_video:
                        episode_frames.append(frame)
                    
                    # 实时显示到 pygame 窗口
                    if pygame_window is not None:
                        pygame_window.show_frame(frame)
            
            done = terminated or truncated
            if episode_length >= 2000:  # 安全上限
                done = True
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        if render_video and episode_frames:
            all_frames.extend(episode_frames)
        
        # 每10个episode汇报一次
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_r = np.mean(rewards)
            avg_l = np.mean(lengths)
            log_callback(
                f"Episode {ep+1}/{num_episodes} | "
                f"Reward: {episode_reward:.1f} | "
                f"Length: {episode_length} | "
                f"AvgReward: {avg_r:.1f} | "
                f"AvgLength: {avg_l:.1f}"
            )
    
    env.close()
    
    # 关闭 pygame 窗口
    if pygame_window is not None:
        pygame_window.close()
    
    results = {
        "rewards": np.array(rewards),
        "lengths": np.array(lengths),
        "frames": all_frames if render_video else None,
        "num_episodes": num_episodes,
        "model_path": model_path,
    }
    
    return results


def save_video(frames, video_path, fps=VIDEO_FPS):
    """将帧列表保存为视频文件"""
    if not frames:
        return False
    
    try:
        import imageio
        # 确保目录存在
        os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else ".", exist_ok=True)
        imageio.mimsave(video_path, frames, fps=fps)
        return True
    except ImportError:
        # 备用方案：使用 OpenCV
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frames[0].shape[:2]
            os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else ".", exist_ok=True)
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            for frame in frames:
                # RGB -> BGR
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                writer.write(frame_bgr)
            writer.release()
            return True
        except ImportError:
            return False


def generate_plots(results, save_dir, log_callback):
    """生成matplotlib图表并保存"""
    rewards = results["rewards"]
    lengths = results["lengths"]
    num_episodes = results["num_episodes"]
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 奖励分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 直方图
    ax = axes[0, 0]
    ax.hist(rewards, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(rewards), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(rewards):.1f}")
    ax.axvline(np.median(rewards), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(rewards):.1f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Reward Distribution (n={num_episodes})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 奖励趋势
    ax = axes[0, 1]
    ax.plot(range(1, num_episodes + 1), rewards, "o-", color="steelblue", markersize=3, linewidth=0.8)
    ax.axhline(np.mean(rewards), color="red", linestyle="--", label=f"Mean: {np.mean(rewards):.1f}")
    ax.fill_between(range(1, num_episodes + 1), np.mean(rewards) - np.std(rewards), np.mean(rewards) + np.std(rewards),
                    alpha=0.2, color="red", label=f"±Std: {np.std(rewards):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward Trend")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 步长分布
    ax = axes[1, 0]
    ax.hist(lengths, bins=20, color="lightcoral", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(lengths), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(lengths):.1f}")
    ax.axvline(np.median(lengths), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(lengths):.1f}")
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Episode Length Distribution (n={num_episodes})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 奖励 vs 步长 散点图
    ax = axes[1, 1]
    ax.scatter(lengths, rewards, alpha=0.6, color="purple", s=30)
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward vs Length Scatter")
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息文本
    stats_text = (
        f"Mean Reward: {np.mean(rewards):.2f}\n"
        f"Std Reward:  {np.std(rewards):.2f}\n"
        f"Min Reward:  {np.min(rewards):.2f}\n"
        f"Max Reward:  {np.max(rewards):.2f}\n"
        f"Mean Length: {np.mean(lengths):.2f}\n"
        f"Std Length:  {np.std(lengths):.2f}"
    )
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    plot_path = os.path.join(save_dir, f"eval_plot_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    log_callback(f"[✓] 图表已保存: {plot_path}")
    
    # 保存原始数据为CSV
    csv_path = os.path.join(save_dir, f"eval_data_{timestamp}.csv")
    np.savetxt(csv_path, np.column_stack([range(1, num_episodes + 1), rewards, lengths]),
               delimiter=",", header="episode,reward,length", comments="")
    log_callback(f"[✓] 数据已保存: {csv_path}")
    
    return plot_path, csv_path


# ------------------- GUI -------------------

class EvalGUI:
    def __init__(self, master):
        self.master = master
        master.title("BipedalWalker 模型评估器")
        master.geometry("650x560")
        master.resizable(False, False)
        
        # 变量
        self.model_path_var = StringVar(value="")
        self.eval_count_var = StringVar(value=str(DEFAULT_EVAL_EPISODES))
        self.use_shaping_var = BooleanVar(value=True)
        self.render_video_var = BooleanVar(value=True)
        self.show_pygame_var = BooleanVar(value=True)
        self.video_path_var = StringVar(value="./eval_results/eval_video.mp4")
        self.plot_path_var = StringVar(value="./eval_results")
        
        self.eval_thread = None
        self.is_running = False
        
        self._build_ui()
    
    def _build_ui(self):
        # 模型选择
        f1 = Frame(self.master)
        f1.pack(pady=8, padx=15, fill="x")
        Label(f1, text="模型路径:", width=12, anchor="w").pack(side="left")
        Entry(f1, textvariable=self.model_path_var, width=45).pack(side="left", padx=5)
        Button(f1, text="浏览...", command=self._browse_model).pack(side="left")
        
        # 评估次数
        f2 = Frame(self.master)
        f2.pack(pady=5, padx=15, fill="x")
        Label(f2, text="评估次数:", width=12, anchor="w").pack(side="left")
        Entry(f2, textvariable=self.eval_count_var, width=10).pack(side="left", padx=5)
        Label(f2, text="次（默认100）", fg="gray").pack(side="left")
        
        # 塑形选项
        f3 = Frame(self.master)
        f3.pack(pady=5, padx=15, fill="x")
        Checkbutton(f3, text="使用奖励塑形（与训练环境一致）", variable=self.use_shaping_var).pack(anchor="w")
        
        # 实时显示选项
        f3b = Frame(self.master)
        f3b.pack(pady=5, padx=15, fill="x")
        Checkbutton(f3b, text="实时显示 pygame 动画窗口（评估时观看）", variable=self.show_pygame_var).pack(anchor="w")
        
        # 录屏选项
        f4 = Frame(self.master)
        f4.pack(pady=5, padx=15, fill="x")
        Checkbutton(f4, text="录屏保存", variable=self.render_video_var).pack(side="left")
        Entry(f4, textvariable=self.video_path_var, width=40).pack(side="left", padx=5)
        Button(f4, text="浏览...", command=self._browse_video_path).pack(side="left")
        
        # 图表保存
        f5 = Frame(self.master)
        f5.pack(pady=5, padx=15, fill="x")
        Label(f5, text="图表/数据保存:", width=12, anchor="w").pack(side="left")
        Entry(f5, textvariable=self.plot_path_var, width=40).pack(side="left", padx=5)
        Button(f5, text="浏览...", command=self._browse_plot_dir).pack(side="left")
        
        # 按钮区
        f6 = Frame(self.master)
        f6.pack(pady=10, padx=15)
        self.start_btn = Button(f6, text="▶ 开始评估", command=self._start_eval, width=15, bg="lightgreen")
        self.start_btn.pack(side="left", padx=5)
        Button(f6, text="■ 停止评估", command=self._stop_eval, width=15, bg="lightcoral").pack(side="left", padx=5)
        
        # 进度
        f7 = Frame(self.master)
        f7.pack(pady=5, padx=15, fill="x")
        self.progress_label = Label(f7, text="就绪", fg="blue")
        self.progress_label.pack(anchor="w")
        
        # 日志区
        f8 = Frame(self.master)
        f8.pack(pady=5, padx=15, fill="both", expand=True)
        Label(f8, text="评估日志:", anchor="w").pack(fill="x")
        self.log_text = scrolledtext.ScrolledText(f8, height=12, width=75, state="disabled")
        self.log_text.pack(fill="both", expand=True)
    
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("ZIP 模型", "*.zip"), ("所有文件", "*.*")],
            initialdir="./models"
        )
        if path:
            self.model_path_var.set(path)
    
    def _browse_video_path(self):
        path = filedialog.asksaveasfilename(
            title="保存视频",
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4")],
            initialdir="./eval_results",
            initialfile="eval_video.mp4"
        )
        if path:
            self.video_path_var.set(path)
    
    def _browse_plot_dir(self):
        path = filedialog.askdirectory(title="选择保存目录", initialdir="./eval_results")
        if path:
            self.plot_path_var.set(path)
    
    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.progress_label.config(text=msg)
    
    def _start_eval(self):
        if self.is_running:
            messagebox.showwarning("警告", "评估正在进行中！")
            return
        
        model_path = self.model_path_var.get().strip()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("错误", "请选择一个有效的模型文件 (.zip)")
            return
        
        try:
            num_episodes = int(self.eval_count_var.get())
            if num_episodes <= 0 or num_episodes > 1000:
                raise ValueError()
        except ValueError:
            messagebox.showerror("错误", "评估次数必须是 1~1000 的整数")
            return
        
        use_shaping = self.use_shaping_var.get()
        render_video = self.render_video_var.get()
        show_pygame = self.show_pygame_var.get()
        video_path = self.video_path_var.get().strip()
        plot_dir = self.plot_path_var.get().strip()
        
        self.is_running = True
        self.start_btn.config(state="disabled", text="评估中...")
        self._log(f"{'='*50}")
        self._log(f"开始评估 | 模型: {os.path.basename(model_path)}")
        self._log(f"评估次数: {num_episodes} | 塑形: {use_shaping} | 录屏: {render_video} | 实时显示: {show_pygame}")
        self._log(f"{'='*50}")
        
        # 在后台线程运行评估
        self.eval_thread = threading.Thread(
            target=self._eval_worker,
            args=(model_path, num_episodes, use_shaping, render_video, show_pygame, video_path, plot_dir),
            daemon=True
        )
        self.eval_thread.start()
    
    def _eval_worker(self, model_path, num_episodes, use_shaping, render_video, show_pygame, video_path, plot_dir):
        try:
            results = evaluate_model(
                model_path=model_path,
                num_episodes=num_episodes,
                use_shaping=use_shaping,
                render_video=render_video,
                video_path=video_path,
                log_callback=self._log,
                show_pygame=show_pygame
            )
            
            if results is None:
                self._log("[✗] 评估失败，请检查模型文件和环境配置")
                self._finish()
                return
            
            # 生成图表
            self._log("[...] 正在生成图表...")
            plot_path, csv_path = generate_plots(results, plot_dir, self._log)
            
            # 保存视频
            if render_video and results.get("frames"):
                self._log("[...] 正在保存视频...")
                success = save_video(results["frames"], video_path)
                if success:
                    self._log(f"[✓] 视频已保存: {video_path}")
                else:
                    self._log("[✗] 视频保存失败（请安装 imageio 或 opencv-python）")
                    self._log("    pip install imageio[ffmpeg] 或 pip install opencv-python")
            
            # 最终统计
            rewards = results["rewards"]
            self._log(f"{'='*50}")
            self._log(f"评估完成！")
            self._log(f"  平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            self._log(f"  中位数:   {np.median(rewards):.2f}")
            self._log(f"  最小:     {np.min(rewards):.2f}")
            self._log(f"  最大:     {np.max(rewards):.2f}")
            self._log(f"  平均步长: {np.mean(results['lengths']):.1f}")
            self._log(f"{'='*50}")
            
            messagebox.showinfo("评估完成", f"平均奖励: {np.mean(rewards):.2f}\n图表已保存到: {plot_dir}")
            
        except Exception as e:
            self._log(f"[✗] 运行时错误: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            self._finish()
    
    def _finish(self):
        self.is_running = False
        self.start_btn.config(state="normal", text="▶ 开始评估")
    
    def _stop_eval(self):
        if not self.is_running:
            return
        self._log("[■] 用户请求停止（当前 episode 结束后退出）")
        # 实际上无法强制终止线程，只能等当前 episode 结束
        messagebox.showinfo("提示", "已发送停止请求，请等待当前 episode 结束后自动退出。")


# ------------------- 主入口 -------------------

def main():
    root = Tk()
    app = EvalGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
