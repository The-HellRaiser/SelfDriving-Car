import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
from stable_baselines3 import PPO      #PPO -> Proximal Policy Optimization
from stable_baselines3.common.evaluation import evaluate_policy  #to evaluate the model 

environment_name = 'CarRacing-v2'    

ppo_path = os.path.join('./best_model.zip')
env = gym.make(environment_name, render_mode="rgb_array")
env = RecordVideo(env, video_folder="../videos_PPO/", episode_trigger=lambda x: True)
model = PPO.load(ppo_path, env=env)
evalue = evaluate_policy(model, env, n_eval_episodes=2, render = True)
env.close()
print(evalue)

