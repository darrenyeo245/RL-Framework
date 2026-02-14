import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MediaEnv(gym.Env):
    """
    Einfaches Beispiel-Environment mit abstraktem Zustand.
    Ziel: Agent bewegt einen "Aktor" (1D) zu einem Ziel (0.0).
    Abstrakte Zustandsfeatures: distance, velocity, priority (fixed).
    Aktionen: 0 = move left, 1 = stay, 2 = move right  (abstract)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=200):
        super().__init__()
        # Abstrakte state: [distance_to_target, velocity, priority]
        # distance in [-2.0, 2.0], velocity in [-1.0, 1.0], priority in [0,1]
        low = np.array([-5.0, -2.0, 0.0], dtype=np.float32)
        high = np.array([5.0, 2.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # discrete abstract actions
        self.action_space = spaces.Discrete(3)  # 0=left, 1=stay, 2=right

        self.state = None
        self.max_steps = max_steps
        self.current_step = 0
        self.dt = 0.1  # time step for velocity updates

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        position = np.random.uniform(-2.0, 2.0)
        velocity = 0.0
        priority = 1.0  # fixed priority for simplicity
        self.state = np.array([position, velocity, priority], dtype=np.float32)
        self.current_step = 0
        return self.state.copy(), {}

    def step(self, action):
        position, velocity, priority = self.state
        # Map abstract action to velocity change
        if action == 0:  # move left
            velocity -= 0.5
        elif action == 2:  # move right
            velocity += 0.5
        else:
            velocity = 0.0  # stay

        # simple physics
        velocity = np.clip(velocity * self.dt, -2.0, 2.0)
        position = np.clip(position + velocity * self.dt, -5.0, 5.0)
        self.state = np.array([position, velocity, priority], dtype=np.float32)

        # reward is negative distance to target (0.0)
        distance = -abs(position - 0.0)
        reward = -distance # want to minimize distance to target
        terminated = False
        truncated = False
        self.current_step += 1
        if distance < 0.05:
            reward += 10.0  # bonus for reaching target
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True
        info = {'distance': distance}
        return self.state.copy(), float(reward), terminated, truncated, info

    def  render(self, mode='human'):
        if mode == 'human':
            if not hasattr(self, 'fig'):
                self.fig, self.ax = plt.subplots(figsize=(10, 2))
                plt.ion()  # Enable interactive mode

            self.ax.clear()

            # Draw the track
            self.ax.plot([-5, 5], [0, 0], 'k-', linewidth=2)

            # Draw the target (green circle at position 0)
            target = patches.Circle((0, 0), 0.2, color='green', alpha=0.5)
            self.ax.add_patch(target)

            # Draw the agent (blue circle)
            position = self.state[0]
            agent = patches.Circle((position, 0), 0.15, color='blue')
            self.ax.add_patch(agent)

            # Add velocity arrow
            velocity = self.state[1]
            if abs(velocity) > 0.01:
                self.ax.arrow(position, 0, velocity * 0.5, 0,
                              head_width=0.1, head_length=0.1, fc='red', ec='red')

            # Set limits and labels
            self.ax.set_xlim(-5.5, 5.5)
            self.ax.set_ylim(-1, 1)
            self.ax.set_xlabel('Position')
            self.ax.set_title(f'Step: {self.current_step} | Distance: {abs(position):.2f}')

            plt.pause(0.01)

        elif mode == 'rgb_array':
            # Return RGB array for video recording
            self.render('human')
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
