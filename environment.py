import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

DRONE_START = np.array([0., 0., 0.], dtype=np.float32)
GOAL_POS    = np.array([2., 0., 2.], dtype=np.float32)

THRUST  = 0.06          #   ← smaller, easier control
DRAG    = 0.92
GOAL_R  = 0.4
MAX_V   = 0.35
MAX_T   = 400           #   max steps / episode

class DroneReach(gym.Env):
    metadata = {"render_mode": None}

    def __init__(self):
        super().__init__()
        hi = np.array([np.inf]*7, dtype=np.float32)
        self.observation_space = Box(-hi, hi, dtype=np.float32)
        self.action_space      = Discrete(6)
        self.state = None
        self.t     = 0

    # -----------------------------------------------------------------
    def _get_obs(self):
        rel = self.pos - GOAL_POS
        vel = self.vel
        dist = np.linalg.norm(rel)
        return np.concatenate([rel, vel, [dist]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = DRONE_START.copy()
        self.vel = np.zeros(3, dtype=np.float32)
        self.t   = 0
        self._prev_dist = np.linalg.norm(self.pos - GOAL_POS)
        return self._get_obs(), {}

    def step(self, action):
        # thrust
        if   action == 0: self.vel[0] -= THRUST
        elif action == 1: self.vel[0] += THRUST
        elif action == 2: self.vel[1] += THRUST
        elif action == 3: self.vel[1] -= THRUST
        elif action == 4: self.vel[2] -= THRUST
        elif action == 5: self.vel[2] += THRUST

        # physics -----------------------------------------------------
        vmag = np.linalg.norm(self.vel)
        if vmag > MAX_V:
            self.vel *= MAX_V / vmag
        self.pos += self.vel
        self.vel *= DRAG
        self.t   += 1

        # reward ------------------------------------------------------
        dist = np.linalg.norm(self.pos - GOAL_POS)
        reward =  1.0 * (self._prev_dist - dist)     # progress
        reward += -0.005                            # time penalty
        done = False

        if dist < GOAL_R:
            reward +=  10.0
            done   = True
        elif self.t >= MAX_T:
            done = True

        self._prev_dist = dist
        return self._get_obs(), reward, done, False, {}

    # -----------------------------------------------------------------
    def render(self):
        pass  