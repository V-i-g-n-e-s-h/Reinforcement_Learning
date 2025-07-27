import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

DEFAULT_OBSTACLES = [
    {
        "min": np.array([0.80, 0.0, 0.80], dtype=np.float32),
        "max": np.array([1.10, 1.5, 1.10], dtype=np.float32),
    },
    {
        "min": np.array([1.80, 0.0, 0.50], dtype=np.float32),
        "max": np.array([2.20, 2.0, 0.80], dtype=np.float32),
    },
    {
        "min": np.array([2.50, 0.0, 2.10], dtype=np.float32),
        "max": np.array([2.90, 1.8, 2.30], dtype=np.float32),
    },
]

DRONE_START = np.array([0., 0., 0.], dtype=np.float32)
GOAL_POS    = np.array([4., 3., 4.], dtype=np.float32)

THRUST  = 0.06     # thrust per action
DRAG    = 0.92
GOAL_R  = 0.4
MAX_V   = 0.35
MAX_T   = 400

class DroneReach(gym.Env):
    metadata = {"render_mode": None}

    def __init__(self, obstacles=None):
        super().__init__()
        hi = np.array([np.inf]*7, dtype=np.float32)
        self.observation_space = Box(-hi, hi, dtype=np.float32)
        self.action_space      = Discrete(6)
        self.obstacles = obstacles if obstacles is not None else DEFAULT_OBSTACLES
        self.pos = None
        self.vel = None
        self.t = 0
        self._prev_dist = None


    def _get_obs(self):
        rel = self.pos - GOAL_POS
        vel = self.vel
        dist = np.linalg.norm(rel)
        return np.concatenate([rel, vel, [dist]]).astype(np.float32)

    def _collided(self, p):
        for ob in self.obstacles:
            if np.all(p >= ob["min"]) and np.all(p <= ob["max"]):
                return True
        return False


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

        if (vmag := np.linalg.norm(self.vel)) > MAX_V:
            self.vel *= MAX_V / vmag
        new_pos = self.pos + self.vel
        self.vel *= DRAG
        self.t += 1

        collided = self._collided(new_pos)
        self.pos = new_pos  # move drone regardless, makes visual clear

        reward = -0.005
        success = False
        done = False

        if collided:
            reward += -5.0
            done   = True
        else:
            dist = np.linalg.norm(self.pos - GOAL_POS)
            reward += (self._prev_dist - dist)        # dense shaping
            if dist < GOAL_R:
                reward += 10.0
                done = True
                success = True
            elif self.t >= MAX_T:
                done = True
            self._prev_dist = dist
        info = {"is_success": success, "collision": collided}
        return self._get_obs(), reward, done, False, info


    def render(self):
        pass  