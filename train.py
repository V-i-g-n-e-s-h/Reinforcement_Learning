import numpy as np
import matplotlib.pyplot as plt
from ursina import *

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,)

from environment import (
    DroneReach,
    DRONE_START,
    GOAL_POS,
    GOAL_R,
    DEFAULT_OBSTACLES,
)

TOTAL_STEPS = 1_000_000
SAVE_PATH   = "drone_ppo_vis"

# ────────────────────────────────────────────────────────────────────
# 1. Ursina scene
# ────────────────────────────────────────────────────────────────────
app = Ursina(borderless=False)

drone_ent = Entity(
        model='cube', 
        color=color.azure,
        scale=(0.3, 0.15, 0.3),
        position=DRONE_START,
    )
goal_ent = Entity(
        model='sphere', 
        color=color.red,
        scale=GOAL_R,
        position=GOAL_POS,
    )

for ob in DEFAULT_OBSTACLES:
    size   = ob["max"] - ob["min"]
    centre = (ob["max"] + ob["min"]) / 2
    Entity(
            model='cube',
            color=color.rgba(0, 255, 0, 100),
            scale=tuple(size),
            position=tuple(centre)
        )

ground = Entity(
        model='plane', 
        scale=20,
        color=color.gray, 
        position=(0, -0.1, 0),
    )

camera.position, camera.rotation_x = (0, 12, -18), 30
fps_text = Text(text='', origin=(-.9,.45))

# ────────────────────────────────────────────────────────────────────
# 2. Gymnasium env subclass providing live visuals
# ────────────────────────────────────────────────────────────────────
class DroneReachVisual(DroneReach):
    def __init__(self, drone_entity, trail=True):
        super().__init__()
        self._drone_entity = drone_entity
        self._trail        = trail
        self._dots         = []

    def step(self, action):
        obs, rew, done, trunc, info = super().step(action)

        self._drone_entity.position = tuple(self.pos)

        if self._trail and len(self._dots) < 4_000:
            self._dots.append(
                    Entity(
                        model='sphere', 
                        scale=.025,
                        color=color.orange,
                        position=self._drone_entity.position
                    )
                )

        return obs, rew, done, trunc, info

class TrainingMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards   = []
        self.lengths   = []
        self.successes = []

    def _on_step(self) -> bool:
        app.step()
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(info['episode']['r'])
                self.lengths.append(info['episode']['l'])
                self.successes.append(info.get('is_success', False))
                if len(self.rewards) % 10 == 0:
                    mean_r = np.mean(self.rewards[-100:])
                    fps_text.text = (
                        f'Eps: {len(self.rewards)}  '
                        f'last R: {self.rewards[-1]:.2f}  '
                        f'100-mean: {mean_r:.2f}'
                    )
        return True

    def _on_training_end(self):
        episodes      = np.arange(1, len(self.rewards) + 1)
        success_rate  = np.cumsum(self.successes) / episodes

        # Plot 1: Reward
        plt.figure(figsize=(10, 7))
        plt.plot(episodes, self.rewards, label='Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('reward_plot.png', dpi=150)
        plt.close()

        # Plot 2: Episode Length
        plt.figure(figsize=(10, 7))
        plt.plot(episodes, self.lengths, label='Episode Length', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.title('Episode Lengths')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('length_plot.png', dpi=150)
        plt.close()

        # Plot 3: Success Rate
        plt.figure(figsize=(10, 7))
        plt.plot(episodes, success_rate, label='Success Rate', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Over Episodes')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('success_rate_plot.png', dpi=150)
        plt.close()


        Text(
            'Training finished!', origin=(0, 0), scale=2,
            color=color.green, background=True
        )
        app.step()

# stop after certain episodes
class StopAfterEpisodes(BaseCallback):
    def __init__(self, max_episodes: int):
        super().__init__()
        self.max_episodes = max_episodes
        self.count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.count += 1
                if self.count >= self.max_episodes:
                    print(f"\033[31m Reached {self.max_episodes} episodes → stopping. \033[0m")
                    return False
        return True

# ────────────────────────────────────────────────────────────────────
# 4. Build learner & train (policy‑gradient loop)
# ────────────────────────────────────────────────────────────────────

def make_env():
    return Monitor(DroneReachVisual(drone_ent, trail=True))

env = DummyVecEnv([make_env])

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    gamma=0.995,
    n_steps=2048,
    batch_size=256,
    verbose=0,
)

print("\033[32m Training with live visualisation - close window to quit early. \033[0m")
try:
    callbacks = CallbackList([
        TrainingMetricsCallback(),
        StopAfterEpisodes(max_episodes=4_000),
    ])
    model.learn(total_timesteps=TOTAL_STEPS,
                callback=callbacks)
finally:
    model.save(SAVE_PATH)
    print(f"\033[32m Model saved to {SAVE_PATH}.zip \033[0m")
    app.userExit()
