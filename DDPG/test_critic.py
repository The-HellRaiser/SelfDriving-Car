

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

from utils import *
from base_sol import *


# Parameters
n_episodes = 5
problem = 'CarRacing-v2'

gym.logger.set_level(40)
all_episode_reward = []

# Initialize simulation
env = gym.make(problem,render_mode = "rgb_array")
env = RecordVideo(env, video_folder="../videos_DDPG/", episode_trigger=lambda x: True)

# env.reset()

# Define custom standard deviation for noise
# We can improve stability of solution, by noise parameters
noise_mean = np.array([0.0, -0.83], dtype=np.float32)
noise_std = np.array([0.0, 4 * 0.02], dtype=np.float32)
solution = BaseSolution(env.action_space, model_outputs=2, noise_mean=noise_mean, noise_std=noise_std)
solution.load_solution('./')


# Loop of episodes
for ie in range(n_episodes):
    state, _ = env.reset()
    solution.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    no_reward_counter = 0

    # One-step-loop
    while not (terminated or truncated):
        env.render()

        action, train_action = solution.get_action(state, add_noise=True)

        # This will make steering much easier
        action /= 4
        new_state, reward, terminated, truncated, info = env.step(action)

        state = new_state
        episode_reward += reward

        if reward < 0:
            no_reward_counter += 1
            if no_reward_counter > 200:
                break
        else:
            no_reward_counter = 0

    all_episode_reward.append(episode_reward)
    average_result = np.array(all_episode_reward[-100:]).mean()
    print('Last result:', episode_reward, 'Average results:', average_result)
