from ursina import *
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

from env_ddpg import (
    DroneReach,
    DRONE_START,
    GOAL_POS,
    GOAL_R,
    DEFAULT_OBSTACLES,
)

app = Ursina()

env = DroneReach()
model = DDPG.load("drone_ddpg_vis", env=DummyVecEnv([DroneReach]))

drone_ent = Entity(model='cube', color=color.azure, scale=(0.4, 0.2, 0.4), position=DRONE_START)
goal_ent  = Entity(model='sphere', color=color.red,   scale=GOAL_R * 2, position=GOAL_POS)

for ob in DEFAULT_OBSTACLES:
    size   = ob['max'] - ob['min']
    centre = (ob['max'] + ob['min']) / 2
    Entity(model='cube', color=color.rgba(0, 255, 0, 100), scale=tuple(size), position=tuple(centre))

ground = Entity(model='plane', scale=20, color=color.gray, position=(0, -0.1, 0))
camera.position, camera.rotation_x = (0,12,-18), 30

state, _ = env.reset()
trail = []

def update():
    global state
    action, _ = model.predict(state, deterministic=True)
    state, _, done, _, info = env.step(action)

    drone_ent.position = tuple(env.pos)
    trail.append(Entity(model='sphere', scale=0.04, color=color.orange, position=drone_ent.position))

    if done:
        if info.get('is_success', False):
            Text('Reached goal!', origin=(0,0), scale=2, color=color.green, background=True)
        else:
            Text('Collision / Timeout!', origin=(0, 0), scale=2, color=color.red, background=True)
        application.pause()

app.run()
