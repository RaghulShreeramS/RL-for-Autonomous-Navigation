import numpy as np
import math

# from gym import Env
# from gym.spaces import Box, Dict
# import gym

from gymnasium import Env
from gymnasium.spaces import Box, Dict
import gymnasium

# from task import CarTask
from dm_control import composer
import cv2
from car import Car
from Car_Task import PressWithSpecificForce
import random
import json
'''
* Things required for a stable baselines env to work:
    - functions which are required are listed below - __init__, step, reset, render, close (these are for framework)
    - For the network, we need observations (images and states), and means to provide actions
    - additional functions - goalreached along with goal distance, obstructed, checkcomplete, reward, goalpos, carpos, getResetObs, getCurrObs

{

    'car/body_pose_2d': array([ 2.45987755e-01,  1.19822533e-06, -1.90241757e-04]), 
    'car/body_position': array([2.45987755e-01, 1.19822533e-06, 4.00422006e-04]), 
    'car/body_rotation': array([ 9.99999994e-01, -6.15834970e-06, -5.92609224e-05, -9.51205127e-05]), 
    'car/body_rotation_matrix': 
    array([ 9.99999975e-01,  1.90241754e-04, -1.18520672e-04,  0.00000000e+00,
       -1.90240294e-04,  9.99999982e-01,  1.23279732e-05,  0.00000000e+00,
        1.18523016e-04, -1.23054255e-05,  9.99999993e-01,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]), 
        'car/body_vel_2d': array([ 7.36842529e-01, -2.09264098e-04]), 
        car/realsense_camera, 
        'car/sensors_acc': array([ 0.92276181,  0.01851117, 10.11273594]), 
        'car/sensors_gyro': array([ 0.00377797, -0.00224604,  0.00382539]), 
        'car/sensors_vel': array([ 7.36844215e-01, -6.92586860e-05,  1.39557987e-02]), 
        'car/steering_pos': array([0.00091535]), 
        'car/steering_vel': array([-3.60766821e-05]), 
        'car/wheel_speeds': array([14.98710914, 14.9969212 , 14.98587148, 15.00220248])
        
}

'''




class CarEnv(Env):

    def __init__(self):

        self.creature = Car()
        self.task = PressWithSpecificForce(self.creature)

        # self.task = CarTask()
        self.original_env = composer.Environment(self.task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

        self.mj_state = self.original_env.reset()
        # print(self.mj_state.observation['car/body_vel_2d'])

        self.endTime = 2048

        self.timeElapsed = 0

        self.reward_func = []

        self.action_space = Box(low = np.array([-1, 0]),
                                high = np.array([1, 1]),
                                shape=(2,), dtype=np.float32)
        
        # pos = self.mj_state.observation['car/body_pose_2d']
        # self.state = np.array([pos[0], pos[1], pos[2], np.sqrt(sum(self.mj_state.observation['car/body_vel_2d'] ** 2)), self.mj_state.observation['car/steering_pos'][0]]) 
        self.state = np.array([np.sqrt(sum(self.mj_state.observation['car/body_vel_2d'] ** 2))]) 
        
        # for i in range(2):
        #     self.state = np.concatenate((self.state, np.array([pos[0], pos[1], pos[2], np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]])))

        self.images = self.getDepthMap()

        # obs_space_low = np.array([-np.inf, -np.inf, -np.pi, -np.inf, -np.inf])
        # obs_space_high = np.array([np.inf, np.inf,  np.pi, np.inf, np.inf])

        obs_space_low = np.array([ -np.inf ])
        obs_space_high = np.array([ np.inf])

        # for i in range(2):
        #     obs_space_low = np.concatenate((obs_space_low, np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])))
        #     obs_space_high = np.concatenate((obs_space_high, np.array([np.inf, np.inf, np.inf, np.inf, np.inf])))
        
        self.observation_space = Dict(
            spaces={
                "vec": Box(low = obs_space_low, high = obs_space_high, shape = (1,), dtype=np.float32),
                "img": Box(low = 0, high = np.inf, shape = self.images.shape, dtype = np.float32),
            }
        )

    

        
       
        self.done = False
      

     


        # self.boxPos = np.array(self.task._button_initial_pose)
        # self.boxPos = self.boxPos[:-1]
        # print("Button position", self.boxPos)
        # print("intial position of the button", self.boxPos)
        # self.init_goal = np.array([10,10]) # self.boxPos # figure this out
        # self.init_car = [0, 0, 0]


        print("Generating environment... Box - ")


    def getDepthMap(self):
        
        image = self.mj_state.observation['car/realsense_camera'].astype(np.float32)
        return image
    
    def step(self, action):

        self.creature.apply_action(self.original_env.physics, action, None)
        self.mj_state = self.original_env.step(action)
        self.timeElapsed += 1
        check_time = self.checkComplete()
        truncated= False
        reward, check = self.task.get_reward(self.original_env.physics)
        

        if check_time == 1:
            self.done = True
        # elif check == 2:
        #     reward += 10000000000
        #     self.done = True
        elif check==3:
            check_time=3
            print("Collision")
            # reward -= 100
            self.done = True
        
        state_obs = self.getCurrObs()
        
        info = {}
        self.rewardAccumulated += reward
        


        if self.done:
            print(self.timeElapsed, "Steps")
            print(check_time, "Check")
            print(self.rewardAccumulated)


        return state_obs, reward, self.done, truncated, info
    
    def getReward(self,action):
       
       reward = 0
       return reward
    
    def getRotation(self, theta):
        return np.array([
            [np.cos(theta), np.sin(theta)],
            [np.sin(theta), -np.cos(theta)]
        ])
    
    def getCurrObs(self):

        # pos = self.mj_state.observation['car/body_pose_2d']
        vel = np.sqrt(sum(self.mj_state.observation['car/body_vel_2d'] ** 2))
        # steer_pos = self.mj_state.observation['car/steering_pos'][0]
        # w, x, y, z = self.mj_state.observation['car/body_rotation']
        # yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))


        # self.state = np.concatenate((np.array([relative_pos[0], relative_pos[1], pos[2], np.sqrt(sum(self.mj_state[3]['car/body_vel_2d'] ** 2)), self.mj_state[3]['car/steering_pos'][0]]), self.state[:-4]))
        # self.state = self.state = np.array([pos[0], pos[1],  pos[2], vel, steer_pos]).astype(np.float32) #, self.state[:-4]))


        self.state = self.state = np.array([ vel]).astype(np.float32) 

        
        obs = self.state

        self.images  = self.getDepthMap()
        
       
        return { "vec": obs, "img": self.images}
    
    
   

    
    # def obstructed(self):
    #     pos = self.mj_state.observation['car/body_pose_2d']
    #     car_pos = [pos[0], pos[1]]
        
    #     physics = self.original_env.physics
    #     is_obstructed = self.task._compute_collision_penalty(physics)
    #     # obs = self.task.getObstacles()
    #     # print(obs)

    #     # print(f"Positions {car_pos}, {obs[1]}" )
    #     # is_obstructed = self.task.in_bounds(car_pos)
    #     # for i in range(obs.shape[0]):
    #     #     dista = np.sqrt((pos[0]-obs[i][0])**2 + (pos[1]-obs[i][1])**2)
    #     #     if dista<0.3:
    #     #         print("Collision Detected!")
                
    #     #         return True
    #     # return False
    #     return  is_obstructed
    
    def goalReached(self):
        dist = self.goalDist()
        return dist <= 0.5

    def goalDist(self):
        pos = self.mj_state.observation['car/body_pose_2d']

        dist = (pos[0]-self.boxPos[0])**2 + (pos[1]-self.boxPos[1])**2
        
        return np.sqrt(dist)
    
    def checkComplete(self):
        
        if self.timeElapsed >= self.endTime: return 1
        # if self.goalReached(): return 2
        # if self.task._compute_collision_penalty(self.original_env.physics)!=0: 
        #     print("Collision detected!")
        #     return 3

        return 0
    
    # See whether we need goal and car pose
    # For now we don't
    def generateRandomBox(self):
        #return np.array([10,0])
        goal_range = [2,10]
        quadrant = np.random.randint(0,2,2)
        if quadrant[0] == 0:
            quadrant[0] = -1
        if quadrant[1] == 0:
            quadrant[1] = -1
        return np.random.randint(goal_range[0], goal_range[1], 2) * quadrant
    
#---------- SEED LOGIC--------------------------------------------------------------------------------------------------------
    # def seed(self, seed=None):
    #     if seed is not None:
    #         random.seed(seed)    

    def reset(self, seed=None, **kwargs):
        # print("Initial Box - ",self.boxPos)
        # print("Initial Car - ", self.init_car)

        # print("Final Distance - ", self.goalDist())
        # print("Final Coordinates - ", self.mj_state[3]['car/body_pose_2d'])

        self.done = False

        self.mj_state = self.original_env.reset()
        # self.boxPos = self.task.goal_graph.current_goal
        
        self.rewardAccumulated = 0
        self.timeElapsed = 0
        observations = self.getCurrObs()
        print("[RESET] Generating env...")
        return observations,{}

    def render(self):
        # pics = np.array(self.mj_state.observation['car/overhead_cam'])
        # cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
  
        # # Using resizeWindow() 
        # cv2.resizeWindow("Resized_Window", 700, 700) 
        
        # # Displaying the image 
        # cv2.imshow("Resized_Window", pics) 
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
        # return
        pass

    def close(self):
        self.original_env.close()