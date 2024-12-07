import os
import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import torch.optim as optim
import torch

import matplotlib.pyplot as plt
import numpy as np


from dm_control.mjcf.physics import Physics
from dm_control import viewer

from Obstacle_avoidance_env import CarEnv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import cv2


class DepthMapFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(DepthMapFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with th.no_grad():
           
            sample_depth = observation_space['img'].sample()[None, :, :, 0]
           
            sample_depth = th.tensor(sample_depth).float()
            sample_depth = self.normalize_depth_map(sample_depth)
            sample_depth = sample_depth.unsqueeze(1)
           
            n_flatten = self.cnn(sample_depth).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
      
        depth_map = observations['img'][:, :, :, 0]
       
        depth_map = self.normalize_depth_map(depth_map)
        depth_map = depth_map.unsqueeze(1)
       
        return self.linear(self.cnn(depth_map))
    
    def normalize_depth_map(self, depth_map):
        min_val = depth_map.min()
        max_val = depth_map.max()
        return (depth_map - min_val) / (max_val - min_val)


def make_env():
    return CarEnv()



def main():
    tb = os.path.join("./Mujoco_learning/Pushr_car_simulation/results/tensor/test_file")
    models = os.path.join("./Mujoco_learning/Pushr_car_simulation/results/models/test_file")
    stats_path = os.path.join("./Mujoco_learning/Pushr_car_simulation/results/logs/test_file")

    num_envs = 8
    env = make_env()
    env = Monitor(env, stats_path)
    env.reset()
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    

    log_path = os.path.join("Car_LocoTransformer-surya-mujoco/Mujoco_learning/Pushr_car_simulation/results/models/Button_File")


    model = PPO(
        policy='MultiInputPolicy',
        n_steps = 2048,
        learning_rate=1e-5,
        n_epochs=50,
        env=env,
        gamma=0.99,
        gae_lambda=0.95,
        batch_size=512,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=1,
        policy_kwargs=dict(
                        features_extractor_class=DepthMapFeatureExtractor,
                        features_extractor_kwargs=dict(features_dim=128),
                    ),
        verbose=0,
        tensorboard_log=tb,
        seed=1,
    )
    # model = SAC(
    #             policy='MultiInputPolicy',
    #             env=env,
    #             batch_size=512,
    #             learning_rate=3e-4,
    #             buffer_size=50000,
    #             learning_starts=10000,
    #             train_freq=4,
    #             gradient_steps=2,
    #             ent_coef='auto',
    #             target_update_interval=10,
    #             gamma=0.99,
    #             tau=0.005,
    #             policy_kwargs=dict(
    #                 features_extractor_class=DepthMapFeatureExtractor,
    #                 features_extractor_kwargs=dict(features_dim=256),
    #             ),
    #             verbose=1,
    #             tensorboard_log=tb,
    #             seed=1,
    #             )

    model.learn(total_timesteps=300000)

    print("Saving end model")
    # model.save(log_path)
    # env.save(stats_path+"test_file"+".pkl")
    ppo_path = os.path.join('Training', 'Saved_Models', 'Button_Task_PPO_10_vel_env_1')
    # new_model.save(ppo_path)
    model.save(ppo_path)


if __name__ == "__main__":
    main()