from ursina import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import (DroneReach, DRONE_START, GOAL_POS, GOAL_R,)


app = Ursina()

env = DroneReach()
model = PPO.load("drone_ppo_vis", env=DummyVecEnv([DroneReach]))

# Ursina scene --------------------------------------------------------
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
        position=(0,-0.1,0),
    )
camera.position, camera.rotation_x = (0,12,-18), 30

state, _ = env.reset()
trail = []

def update():
    global state
    action, _ = model.predict(state, deterministic=True)
    state, _, done, _, _ = env.step(int(action))

    drone_ent.position = tuple(env.pos)
    trail.append(Entity(model='sphere', scale=0.04,
                        color=color.orange, position=drone_ent.position))

    if done:
        Text('Reached goal!', origin=(0,0), scale=2, color=color.green, background=True)
        application.pause()

app.run()
