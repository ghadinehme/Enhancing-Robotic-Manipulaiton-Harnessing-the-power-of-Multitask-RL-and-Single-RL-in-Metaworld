import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import json


'''
To define the class for Replay Buffer
# '''
# class ReplayBuffer():
#     def __init__(self, max_size, input_shape, n_actions):
#         '''
#         max_size = maximum size of the replay buffer
#         input_shape = observation space
#         n_actions = # actinos the agent can take

#         to store everything in individual arrays
#         '''
#         self.mem_size = max_size
#         self.mem_cntr = 0

#         # to store the states 
#         self.state_memory = np.zeros((self.mem_size, *input_shape))

#         # to store the states that we get after taking an action
#         self.new_state_memory = np.zeros((self.mem_size, *input_shape))

#         self.action_memory = np.zeros((self.mem_size, n_actions))
#         self.reward_memory = np.zeros(self.mem_size)

#         # to store the "done" commands (the terminal flags)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    
#     def store_transition(self, state, action, reward, state_, done):
#         '''
#         To store the transitions, "state_" stands for the next state
#         '''
#         index = self.mem_cntr % self.mem_size

#         #print("Inside store_transition function...")

#         self.state_memory[index] = state
#         self.new_state_memory[index] = state_
#         self.action_memory[index] = action
#         self.reward_memory[index] = reward
#         self.terminal_memory[index] = done

#         self.mem_cntr += 1

    
#     def sample_buffer(self, batch_size):
#         '''
#         To sample from the buffer
#         '''
#         max_mem = min(self.mem_cntr, self.mem_size)

#         # to sample
#         batch = np.random.choice(max_mem, batch_size)

#         states = self.state_memory[batch]
#         states_ = self.new_state_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         dones = self.terminal_memory[batch]

#         return states, actions, rewards, states_, dones
    
    
#     def save_buffer(self):

#         # Create a dictionary to store the numpy arrays with their labels
#         arrays_dict = {
#             "self.state_memory": self.state_memory.tolist(),
#             "self.new_state_memory": self.new_state_memory.tolist(),
#             "self.action_memory": self.action_memory.tolist(),
#             "self.reward_memory": self.reward_memory.tolist(),
#             "self.terminal_memory": self.terminal_memory.tolist()
#         }

#         # Convert the dictionary to a JSON string
#         json_string = json.dumps(arrays_dict)

#         # Save the JSON string to a file
#         with open("pick_place_replay_buffer.json", "w") as f:
#             f.write(json_string)    




# ''' 
# To define the critic network

# Outputs the value of Q(s,a); same as we did in the assignment
# '''





class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        '''
        max_size = maximum size of the replay buffer
        input_shape = observation space
        n_actions = # actinos the agent can take

        to store everything in individual arrays
        '''
        self.mem_size = max_size
        self.mem_cntr = 0

        # to store the states 
        self.state_memory = np.zeros((self.mem_size, *input_shape))

        # to store the states that we get after taking an action
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))

        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

        # to store the "done" commands (the terminal flags)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    
    def store_transition(self, state, action, reward, state_, done):
        '''
        To store the transitions, "state_" stands for the next state
        '''
        index = self.mem_cntr % self.mem_size

        #print("Inside store_transition function...")

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    
    def sample_buffer(self, batch_size):
        '''
        To sample from the buffer
        '''
        max_mem = min(self.mem_cntr, self.mem_size)

        # to sample
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
    
    def import_buffer(self, files):
        # Import replay buffer from json file
        print('Importing the buffer')
        state_memory = []
        new_state_memory = []
        action_memory = []
        reward_memory = []
        terminal_memory = []

        for file in files:
            path = os.path.dirname(os.path.abspath(__file__)) + file
            with open(path, "r") as f:
                json_string = f.read()
                data = json.loads(json_string)

            state_memory += data["self.state_memory"]
            new_state_memory += data["self.new_state_memory"]
            action_memory += data["self.action_memory"]
            reward_memory += data["self.reward_memory"]
            terminal_memory += data["self.terminal_memory"]

        # to store the states 
        self.state_memory = np.array(state_memory)

        # to store the states that we get after taking an action
        self.new_state_memory = np.array(new_state_memory)

        self.action_memory = np.array(action_memory)
        self.reward_memory = np.array(reward_memory)

        # to store the "done" commands (the terminal flags)
        self.terminal_memory = np.array(terminal_memory)
        

    
    def save_buffer(self):

        # Create a dictionary to store the numpy arrays with their labels
        arrays_dict = {
            "self.state_memory": self.state_memory.tolist(),
            "self.new_state_memory": self.new_state_memory.tolist(),
            "self.action_memory": self.action_memory.tolist(),
            "self.reward_memory": self.reward_memory.tolist(),
            "self.terminal_memory": self.terminal_memory.tolist()
        }

        # Convert the dictionary to a JSON string
        json_string = json.dumps(arrays_dict)

        # Save the JSON string to a file
        with open("mtsac_replay_buffer.json", "w") as f:
            f.write(json_string)    



