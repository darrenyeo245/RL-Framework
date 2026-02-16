import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.media_env import MediaEnv
import os


def train(out_dir='models', total_timesteps=100000, render=False):
    os.makedirs(out_dir, exist_ok=True)
    env = MediaEnv()
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tb_logs/")
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=out_dir, name_prefix='rl_model')

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(os.path.join(out_dir, 'final_model'))
    print("Training completed and model saved.")

    # Visualize after training
    if render:
        visualize_trained_model(model, env)

    return model, env


def visualize_trained_model(model, env, num_episodes=3):
    """Visualize the trained agent"""
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    train(out_dir='models', total_timesteps=10000, render=True)