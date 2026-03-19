import gymnasium as gym
import numpy as np

class MediaEnv(gym.Env):
    def __init__(self, osc_interface):
        super().__init__()

        self.osc = osc_interface

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def step(self, action):
        # Action senden
        self.osc.send_action(action)

        # Neuen Zustand holen (letzter empfangener Zustand)
        state = self.osc.get_state()

        # Auf manuelle Reward-Eingabe warten
        reward = self.osc.get_reward(wait_for_new=True)

        terminated = False
        truncated = False

        return state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=float), {}
