"""
Main Loop of simulation.
Here a problem and a solution are defined.
"""

import gymnasium as gym
import numpy as np

from driver_critic.utils import *
from driver_critic.base_sol import *
import warnings
import time
warnings.filterwarnings("ignore")



# Parameters
n_episodes = 5000
problem = 'CarRacing-v2'

gym.logger.set_level(40)
preview = False
best_result = 0
all_episode_reward = []

# Initialize simulation
env = gym.make(problem,render_mode = None)
env.reset()
# env.viewer.window.on_key_press = key_press
# env.viewer.window.on_key_release = key_release

# Define custom standard deviation for noise
# We need lest noise for steering
noise_std = np.array([0.1, 4 * 0.2], dtype=np.float32)
solution = BaseSolution(env.action_space, model_outputs=2, noise_std=noise_std)

episode_count = 1
start_time = time.time()
# Loop of episodes
for ie in range(n_episodes):
    ep_start_time = time.time()
    state, _ = env.reset()
    solution.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    no_reward_counter = 0

    # One-step-loop
    while not (terminated or truncated):
        if preview:
            env.render()

        action, train_action = solution.get_action(state)

        # This will make steering much easier
        action /= 4
        new_state, reward, terminated, truncated, info = env.step(action)

        # Models action output has a different shape for this problem
        solution.learn(state, train_action, reward, new_state)
        state = new_state
        episode_reward += reward

        if reward < 0:
            no_reward_counter += 1
            if no_reward_counter > 200:
                break
        else:
            no_reward_counter = 0

    all_episode_reward.append(episode_reward)
    average_result = np.array(all_episode_reward[-10:]).mean()
    ep_end_time = time.time()
    print(f'Episode {episode_count} result:', episode_reward, 'Average results:', average_result, "Time Taken:",((ep_end_time-ep_start_time)/60))
    episode_count+= 1

    if episode_reward > best_result:
        print('Saving best solution')
        solution.save_solution()
        best_result = episode_reward
    
end_time = time.time()

print("Total Time taken to Complete 5000 episodes is :", ((end_time-start_time)/60))
