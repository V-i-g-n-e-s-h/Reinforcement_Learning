import numpy as np
from ursina import *

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,)

from environment import DroneReach, DRONE_START, GOAL_POS, GOAL_R

TOTAL_STEPS = 1_000_000      # ~10-15 min CPU with render on
SAVE_PATH   = "drone_ppo_vis"

# ────────────────────────────────────────────────────────────────────
# 1. Ursina scene
# ────────────────────────────────────────────────────────────────────
app = Ursina(borderless=False)

drone_ent = Entity(
        model='cube', 
        color=color.azure,
        scale=(0.4,0.2,0.4), 
        position=DRONE_START,
    )
goal_ent = Entity(
        model='sphere', 
        color=color.red,
        scale=GOAL_R*2, 
        position=GOAL_POS,
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
# 2. Gymnasium env subclass that drives the Ursina entity           ▲
# ────────────────────────────────────────────────────────────────────
class DroneReachVisual(DroneReach):
    """ Same physics, but updates Ursina every step. """
    def __init__(self, drone_entity, trail=True):
        super().__init__()
        self._drone_entity = drone_entity
        self._trail        = trail
        self._dots         = []

    def step(self, action):
        obs, rew, done, trunc, info = super().step(action)

        # update 3-D model
        self._drone_entity.position = tuple(self.pos)

        if self._trail and len(self._dots) < 4000:
            self._dots.append(
                    Entity(
                        model='sphere', 
                        scale=.025,
                        color=color.orange,
                        position=self._drone_entity.position
                    )
                )

        return obs, rew, done, trunc, info


# ────────────────────────────────────────────────────────────────────
# 3. SB-3 Callback that pumps one Ursina frame per learner step     ▲
# ────────────────────────────────────────────────────────────────────
class UrsinaRenderCallback(BaseCallback):
    """Keeps the window responsive & shows FPS and ep rewards."""
    def __init__(self):
        super().__init__()
        self.ep_rewards = []

    def _on_step(self) -> bool:
        app.step()                         # pump render loop
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:          # pulled from Monitor
                self.ep_rewards.append(info["episode"]["r"])
                if len(self.ep_rewards) % 10 == 0:
                    mean = np.mean(self.ep_rewards[-100:])
                    fps_text.text = (
                            f'episodes: {len(self.ep_rewards)}  '
                            f'last R: {self.ep_rewards[-1]:.2f}  '
                            f'100-mean: {mean:.2f}'
                        )
        return True

    def _on_training_end(self):
        Text(
                'Training finished!', 
                origin=(0,0),
                scale=2, 
                color=color.green, 
                background=True,
            )
        app.step()  # draw the text once

class StopAfterEpisodes(BaseCallback):
    def __init__(self, max_episodes: int):
        super().__init__()
        self.max_episodes = max_episodes
        self.count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.count += 1
                if self.count >= self.max_episodes:
                    print(f"🔴 Reached {self.max_episodes} episodes → stopping.")
                    return False   # stops training
        return True

# ────────────────────────────────────────────────────────────────────
# 4. Build env → learner → train (with render callback)             ▲
# ────────────────────────────────────────────────────────────────────
def make_env():
    """Factory needed by DummyVecEnv."""
    return Monitor(DroneReachVisual(drone_ent, trail=True))

env = DummyVecEnv([make_env])        # vectorised, but single env

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.995,
    n_steps=2048,
    batch_size=256,
    verbose=0,
)

print("🟢  Training with live visualisation - close window to quit early.")
try:
    callbacks = CallbackList([
        UrsinaRenderCallback(),
        StopAfterEpisodes(max_episodes=4_000),
    ])
    model.learn(total_timesteps=TOTAL_STEPS,
                callback=callbacks)
finally:
    model.save(SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}.zip")
    app.userExit()                   # closes the window if still open
