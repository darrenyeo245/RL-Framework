import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MediaEnv(gym.Env):
    """
    Tracking environment: Agent follows a moving target.
    The closer the agent stays to the target, the higher the reward.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=200):
        super().__init__()
        self.max_steps = max_steps
        self.dt = 0.1

        # State: [agent_position, agent_velocity, target_position, target_velocity, distance, time_normalized]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -2.0, -5.0, -2.0, 0.0, 0.0]),
            high=np.array([5.0, 2.0, 5.0, 2.0, 10.0, 1.0]),
            dtype=np.float64
        )

        # Action: 0=left, 1=stay, 2=right
        self.action_space = spaces.Discrete(3)

        self.state = None
        self.current_step = 0
        self.agent_position = 0.0
        self.agent_velocity = 0.0
        self.target_position = 0.0
        self.target_velocity = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Random start positions
        self.agent_position = np.random.uniform(-2.0, 2.0)
        self.agent_velocity = 0.0

        # Target starts at random position with random velocity
        self.target_position = np.random.uniform(-2.0, 2.0)
        self.target_velocity = np.random.uniform(-0.5, 0.5)

        distance = abs(self.agent_position - self.target_position)
        self.state = np.array([
            self.agent_position,
            self.agent_velocity,
            self.target_position,
            self.target_velocity,
            distance,
            0.0  # time_normalized
        ], dtype=np.float32)

        self.current_step = 0
        return self.state.copy(), {}

    def step(self, action):
        # Update agent velocity based on action
        if action == 0:  # move left
            self.agent_velocity -= 0.5
        elif action == 2:  # move right
            self.agent_velocity += 0.5
        else:  # stay (action == 1)
            self.agent_velocity *= 0.5  # damping

        # Clip agent velocity
        self.agent_velocity = np.clip(self.agent_velocity, -2.0, 2.0)

        # Update agent position
        self.agent_position += self.agent_velocity * self.dt
        self.agent_position = np.clip(self.agent_position, -5.0, 5.0)

        # Update target movement (sinusoidal pattern + random walk)
        # Target changes direction occasionally
        if np.random.random() < 0.05:  # 5% chance to change direction
            self.target_velocity = np.random.uniform(-0.8, 0.8)

        # Target follows sinusoidal pattern
        time_factor = self.current_step * 0.02
        self.target_velocity += 0.3 * np.sin(time_factor)
        self.target_velocity = np.clip(self.target_velocity, -1.0, 1.0)

        # Update target position
        self.target_position += self.target_velocity * self.dt
        self.target_position = np.clip(self.target_position, -5.0, 5.0)

        # Bounce target off boundaries
        if self.target_position >= 4.9 or self.target_position <= -4.9:
            self.target_velocity *= -1

        # Calculate distance
        distance = abs(self.agent_position - self.target_position)

        # Update state
        time_normalized = self.current_step / self.max_steps
        self.state = np.array([
            self.agent_position,
            self.agent_velocity,
            self.target_position,
            self.target_velocity,
            distance,
            time_normalized
        ], dtype=np.float32)

        # Calculate reward based on distance (closer = higher reward)
        # Exponential reward: closer distances give exponentially higher rewards
        distance_reward = np.exp(-2.0 * distance)  # Range: ~0 to 1

        # Bonus for being very close
        if distance < 0.2:
            distance_reward += 2.0
        elif distance < 0.5:
            distance_reward += 1.0

        reward = distance_reward

        # Check termination (optional: terminate if very close for long time)
        self.current_step += 1
        terminated = False  # No early termination, agent should keep tracking
        truncated = self.current_step >= self.max_steps

        info = {
            'distance': float(distance),
            'agent_position': float(self.agent_position),
            'target_position': float(self.target_position)
        }

        return self.state.copy(), float(reward), terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            if not hasattr(self, 'fig'):
                self.fig, self.ax = plt.subplots(figsize=(12, 4))
                plt.ion()

            self.ax.clear()

            # Draw track
            self.ax.plot([-5, 5], [0, 0], 'k-', linewidth=2)

            # Draw target (moving actor)
            target = patches.Circle((self.target_position, 0), 0.2,
                                    color='red', alpha=0.6, label='Target')
            self.ax.add_patch(target)
            # Target velocity indicator
            self.ax.arrow(self.target_position, 0,
                          self.target_velocity * 0.5, 0,
                          head_width=0.1, head_length=0.1,
                          fc='red', ec='red', alpha=0.3)

            # Draw agent
            agent = patches.Circle((self.agent_position, 0), 0.15,
                                   color='blue', alpha=0.8, label='Agent')
            self.ax.add_patch(agent)
            # Agent velocity indicator
            self.ax.arrow(self.agent_position, 0,
                          self.agent_velocity * 0.5, 0,
                          head_width=0.1, head_length=0.1,
                          fc='blue', ec='blue', alpha=0.3)

            # Draw connection line
            self.ax.plot([self.agent_position, self.target_position],
                         [0, 0], 'g--', alpha=0.5, linewidth=1)

            distance = abs(self.agent_position - self.target_position)
            self.ax.set_xlim(-5.5, 5.5)
            self.ax.set_ylim(-1.5, 1.5)
            self.ax.set_xlabel('Position')
            self.ax.set_title(
                f'Step: {self.current_step} | Distance: {distance:.2f}')
            self.ax.legend(loc='upper right')

            plt.pause(0.01)

        elif mode == 'rgb_array':
            self.render('human')
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data
        return None

    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
