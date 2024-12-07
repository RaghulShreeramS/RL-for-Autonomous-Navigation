from Obstacle_avoidance_env import CarEnv

import car
from dm_control import viewer
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC

from Car_Task import PressWithSpecificForce
import os
import numpy as np

env=CarEnv()
from dm_control import composer
creature = car.Car()
task = PressWithSpecificForce(creature)
original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

state,_=env.reset()

ppo_path = "/home/shreeram/Desktop/123/Car_LocoTransformer-surya-mujoco/Mujoco_learning/Pushr_car_simulation/Training/Saved_Models/Button_Task_PPO_10_vel_env_1.zip"
model = PPO.load(ppo_path)

def random_policy(time_step):
   

    # pos = time_step.observation['car/body_pose_2d']
    image = time_step.observation['car/realsense_camera']
    obs = (np.array([np.sqrt(sum(time_step.observation['car/body_vel_2d'] ** 2))]))

    state = { "vec": obs, "img": image}

    action, _=model.predict(state,deterministic=True)
    # print(action)
    state, reward, done, _, info=env.step(action)
    return action

viewer.launch(original_env, policy=lambda timestep: random_policy(timestep))
