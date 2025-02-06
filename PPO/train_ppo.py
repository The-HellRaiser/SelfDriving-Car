import gymnasium as gym
import os
from stable_baselines3 import PPO      #PPO -> Proximal Policy Optimization
from stable_baselines3.common.vec_env import DummyVecEnv 
from stable_baselines3.common.callbacks import EvalCallback


environment_name = 'CarRacing-v2'    
# env = gym.make(environment_name,render_mode = "rgb_array")


env = gym.make(environment_name,render_mode = "rgb_array")
env = DummyVecEnv([lambda: env])

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path,
            learning_rate=0.001,  # Increase the learning rate
            ent_coef=0.01)        # Adjust the entropy coefficient
ppo_path = os.path.join('./best_model/')
eval_env = model.get_env()
eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=ppo_path,
                             n_eval_episodes=8,
                             eval_freq=5000,verbose=1,
                             deterministic=True, render=False)
model.learn(total_timesteps=500000,callback=eval_callback)
ppo_path = os.path.join('./PPO_2m_Model_final')
model.save(ppo_path)