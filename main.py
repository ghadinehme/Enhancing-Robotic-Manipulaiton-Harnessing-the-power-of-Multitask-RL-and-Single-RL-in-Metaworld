'''
I have changed the files of "mujoco_env.py", "swayer_xyz_env.py". So make sure to check those files. 

I am surpassing the "MjViewer class" of mujoco_py library and instead using "MjRenderContextOffscreen" and the 
"_get_viewer" function in the mujoco_env class is not being used anymore.

'''




from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
import sys, time,os
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import cv2
import numpy as np
import matplotlib.pyplot as plt

from network_replay_buff import ReplayBuffer
from SAC import Agent, ActorNetwork, CriticNetwork, ValueNetwork

sys.path.append("/home/tejas/Documents/Stanford/CS 224R/Final Project/Metaworld/metaworld/envs/mujoco/")

#from metaworld.envs.mujoco.mujoco_env import mujoco_env
from mujoco_env import MujocoEnv


import pygame
from pygame.locals import QUIT, KEYDOWN
pygame.init()
screen = pygame.display.set_mode((400, 300))



# env = SawyerPickPlaceEnvV2()
# print("Max path length: ", env.max_path_length)
# sys.exit()

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)





def main():

    env = SawyerPickPlaceEnvV2()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.reset()
    env._freeze_rand_vec = True
    lock_action = False
    random_action = False
    # obs = env.reset()
    # action = np.zeros(4)

    agent = Agent(input_dims=env.observation_space.shape, env = env, n_actions = env.action_space.shape[0])


    n_episodes = 20

    score_history = []
    num_episodes = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render()
    
    

    for i in range(n_episodes):
        observation = env.reset()
        #env.render()
        done = False
        score = 0
        counter = 0

        while not done:
            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            # print(info)
            # print()

            time.sleep(0.2)
            score += reward

            agent.remember(observation, action, reward, observation_, done)

            if not load_checkpoint:
                agent.learn()

            observation = observation_
            counter = counter + 1

            if counter == env.max_path_length - 1:
                break


        score_history.append(score)
        num_episodes.append(i)
        avg_score = np.mean(score_history[-100:])

        # if avg_score > best_score:
        #     best_score = avg_score
        #     if not load_checkpoint:
        #         agent.save_models()

        print('episode = ', i, ' score = %.1f' % score, 'avg_score = %.1f' % avg_score)
        print("="*100)


    
    # to save the trained models
    if not load_checkpoint:
        agent.save_models()
    
    # to save the replay buffer
    if not load_checkpoint:
        print("Saving replay buffer....")
        agent.memory.save_buffer()
        print('Replay buffer has been saved....')
    
    
    # to save the number of episodes and the corresponding episode rewards in np arrays
    reward_filename = "per_episode_reward_history.npy"
    episode_filename = 'episodes.npy'

    score_history = np.array(score_history)
    num_episodes = np.array(num_episodes)

    print("Saving the numpy arrays....")

    np.save(reward_filename, score_history)
    np.save(episode_filename, num_episodes)

    print("Saved the numpy arrays....")




main()





# char_to_action = {
#     'w': np.array([0, -1, 0, 0]),
#     'a': np.array([1, 0, 0, 0]),
#     's': np.array([0, 1, 0, 0]),
#     'd': np.array([-1, 0, 0, 0]),
#     'q': np.array([1, -1, 0, 0]),
#     'e': np.array([-1, -1, 0, 0]),
#     'z': np.array([1, 1, 0, 0]),
#     'c': np.array([-1, 1, 0, 0]),
#     'k': np.array([0, 0, 1, 0]),
#     'j': np.array([0, 0, -1, 0]),
#     'h': 'close',
#     'l': 'open',
#     'x': 'toggle',
#     'r': 'reset',
#     'p': 'put obj in hand',
# }



# num_episodes = 1000

# '''
# The code below is just ot take random actions
# '''

# counter = 0

# while True:
#     done = False
#     if not lock_action:
#         action[:3] = 0
#     if random_action:
#         for event in pygame.event.get():
#             event_happened = True
#             if event.type == QUIT:
#                 sys.exit()
#             if event.type == KEYDOWN:
#                 char = event.dict['key']
#                 new_action = char_to_action.get(chr(char), None)
#                 if new_action == 'toggle':
#                     lock_action = not lock_action
#                 elif new_action == 'reset':
#                     done = True
#                 elif new_action == 'close':
#                     action[3] = 1
#                 elif new_action == 'open':
#                     action[3] = -1
#                 elif new_action is not None:
#                     action[:3] = new_action[:3]
#                 else:
#                     action = np.zeros(3)
#                 print(action)
#     else:
#         action = env.action_space.sample()

    
#     '''
#     You are getting the RGB image and also the 39 dimensional vector in "ob" along with reward, done, and info. 

#     Info is a dictionary with the following structure

#             info = {
#             'success': success,
#             'near_object': near_object,
#             'grasp_success': grasp_success,
#             'grasp_reward': grasp_reward,
#             'in_place_reward': in_place_reward,
#             'obj_to_target': obj_to_target,
#             'unscaled_reward': reward
#         }

#     '''
#     img, ob, reward, done, info = env.step(action)


#     print("Took action: ", action)
#     counter = counter + 1

#     org_path = r"/home/tejas/Documents/Stanford/CS 224R/Final Project/Metaworld/images/color"
#     save_path = os.path.join(org_path, '{}'.format(counter) + ".png")
    

#     cv2.imwrite(save_path, img)

#     time.sleep(0.2)
