# RL for Autonomous Navigation

This repository contains code and experiments for developing reinforcement learning algorithms to enable autonomous navigation for an F1TENTH car in obstacle-laden environments.

## Features
- Developed a reinforcement learning framework in MuJoCo, where an F1tenth car navigates through randomly placed obstacles. The simulation environment incorporated both visual inputs from a Intel Real Sense stereo camera and state observables.
- Designed and implemented a MultiInputPolicy using CNN for visual inputs and MLP for state inputs. Trained the model using Proximal Policy Optimization (PPO) algorithm to predict steering angles and throttle for the car.
- Achieved successful navigation after 500K timesteps by parallelizing learning across 8 environments, designing a tailored reward function and fine-tuning hyperparameters for optimal performance.
- Trained and deployed a PyTorch model by converting it to TensorRT with ONNX, building ROS nodes, and executing on the Jetson TX2 GPU using CUDA, enabling successful navigation in the F1tenth car.

## Successful Navigation of F1Tenth car Video
- Click this [Video link](https://youtu.be/Q55vnLB0K6M) to see the Autonomous Navigation of a RL trained F1Tenth car
