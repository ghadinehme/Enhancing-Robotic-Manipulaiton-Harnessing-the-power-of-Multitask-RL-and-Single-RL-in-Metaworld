import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from network_replay_buff import ReplayBuffer


import sys

class Agent():

    '''
    reward_scale - Reward scaling is an imp hyperparameter to consider as it drives the entropy
    Tau - factor by which we are going to modulate the factors of our target value network (there is a value
    network and target value network. So instead of a hard copy, we are doing a soft copy by detuning the 
    parameters)

    need to change the actions and the reward scale
    '''
    def __init__(self, task_encoding = 'one-hot', alpha=0.00003, beta=0.0003, input_dims=[46,],
            env=None, gamma=0.99, n_actions=4, max_size=10, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=8, reward_scale=2):
        
        try: 
            maximum = env.action_space.high
        except:
            maximum = env._action_spec.maximum


        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.use_tb = True
        
        
        
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action= maximum )
        

        
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1')
        
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2')
        
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    
    def choose_action(self, observation):
        state = torch.Tensor(observation).view(1,-1).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    
    '''
    to soft update the parameters of the target value network wrt to the value network
    '''
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()


    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()



    def act(self):
        pass




    # Adapt to evaluation mode to return the mean of the actions
    def learn(self, trans_tuple=None, step =None):
        # go back to the program if we do not have sufficient transitions i.e. sufficient data in the replay buffer
        
        metrics = dict()
        
        if trans_tuple == None:
            # print('here')
            # print(self.memory.mem_cntr, self.batch_size)
            if self.memory.mem_cntr < self.batch_size:
                return

            # print("Collected enough data and have started learning...")
            # print("="*50)
            
            # to sample our buffer
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            # print("Original data")
            # for i in (state, action, reward, new_state, done): 
            #     print(i.shape)



        else: 

            self.use_tb = True
            
            state, action, reward, discount, new_state, step_type, next_step_type = trans_tuple

            # print("Original data for SLRL")

            

            # for i in trans_tuple: ssss
            #     print(i.shape)


            done = next_step_type.clone()
            done[done < 2] = 1
            done[done == 2] = 0

            if self.use_tb:
                metrics['batch_reward'] = reward.mean().item()

        # to transform numpy arrays to pytorch tensors
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)  # these are the actions sampled fro the replay buffer

        
        # to calculate the value current state and next state via the value and target value networks
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        # to set the terminal states value to be 0
        value_[done] = 0.0

        # to get the actions according to the new policy wihtout using the reparameterization trick
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)





        # to critic values under the new policy. NOTE: Using"actions" and not "action"
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)  # to overcome overestimation
        critic_value = critic_value.view(-1)

        # to define the VALUE NETWORK LOSS
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()


        
        # to define the ACTOR NETWORK loss
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_probs.mean().item()
            metrics['alpha_value'] = 1




        # to define the CRITIC NETWORK LOSS
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # to add the entropy term so as to encourage exploration
        q_hat = self.scale*reward.view(-1,) + self.gamma*value_

        # NOTE: using the action from the replay buffer
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # to update the target value network parameters
        self.update_network_parameters()
        
        if self.use_tb:
            metrics['critic_target_q'] = critic_1_loss.mean().item()
            metrics['critic_q1'] = q1_old_policy.mean().item()
            metrics['critic_q2'] = q2_old_policy.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        if trans_tuple !=None:
            return metrics
        


     
class CriticNetwork(nn.Module):

    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='/tmp/sac'):
         
        '''
        beta - it is the learning rate
        input dims - # input dimensions (obs)
        '''
        
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # directly passing the state and action pair together
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # output layer - outputting the value of Q(s,a) and so in a scalar value
        self.q = nn.Linear(self.fc2_dims, 1) 

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # to transfer everything to GPU
        self.to(self.device)

    
    def forward(self, state, action):
        # along the batch dimension
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self):
        
        # print(self.checkpoint_file)
        path = os.path.dirname(os.path.abspath(__file__)) + self.checkpoint_file + '.zip'
        self.load_state_dict(torch.load(path))


class ValueNetwork(nn.Module):

    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='/tmp/sac'):
        
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)

        # outputting a scalar quantity
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self):
       
        print(self.checkpoint_file)
        path = os.path.dirname(os.path.abspath(__file__)) + self.checkpoint_file + '.zip'
        self.load_state_dict(torch.load(path))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=4, name='actor', chkpt_dir='/tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        '''
        this is the scaling factor for the action space as we have tanh as the output. but the range of actions in the metaworld is btw
        -1 and 1 so we are not using this max_action variable or setting it to 1
        '''
        self.max_action = max_action

        # to avoid log or division by 0
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # mean of the distribution for the policy; equal to the number of actions that we can take
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        # to define the standard deviation
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        # to predict the mean and std using the same inpui.e. output of fc2
        mu = self.mu(prob) 
        sigma = self.sigma(prob)

        # to prevent the distribution from getting too big
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        # to sample the actions from the distribution
        if reparameterize:
            # if we want to add more noise into the sampled action
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        #action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        # this is to take the action in the env 
        action = torch.tanh(actions).to(self.device)

        # to calculate the loss wrt to the sampled action
        log_probs = probabilities.log_prob(actions)

        # to avoid dividing by 0
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    # def eval(self):
        
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self):

        # print(self.checkpoint_file)
        path = os.path.dirname(os.path.abspath(__file__)) + self.checkpoint_file + '.zip'
        self.load_state_dict(torch.load(path))

        # self.load_state_dict(torch.load(sys.path[0] + self.checkpoint_file +'.zip'))







class AgentEmbed():

    '''
    reward_scale - Reward scaling is an imp hyperparameter to consider as it drives the entropy
    Tau - factor by which we are going to modulate the factors of our target value network (there is a value
    network and target value network. So instead of a hard copy, we are doing a soft copy by detuning the 
    parameters)

    need to change the actions and the reward scale
    '''
    def __init__(self, task_encoding = 'one-hot', alpha=0.00003, beta=0.0003, input_dims=[46,],
            env=None, gamma=0.99, n_actions=4, max_size=10, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=8, reward_scale=2):
        
        try: 
            maximum = env.action_space.high
        except:
            maximum = env._action_spec.maximum


        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.use_tb = True
        
        print('Input shape is as follows', input_dims)
        
        self.actor = ActorNetworkEmbed(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action= maximum )
        

        
        self.critic_1 = CriticNetworkEmbed(beta, input_dims, n_actions=n_actions,
                    name='critic_1')
        
        self.critic_2 = CriticNetworkEmbed(beta, input_dims, n_actions=n_actions,
                    name='critic_2')
        
        self.value = ValueNetworkEmbed(beta, input_dims, name='value')
        self.target_value = ValueNetworkEmbed(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    
    def choose_action(self, observation):
        state = torch.Tensor(observation).view(1,-1).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    
    '''
    to soft update the parameters of the target value network wrt to the value network
    '''
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()


    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()



    def act(self):
        pass




    # Adapt to evaluation mode to return the mean of the actions
    def learn(self, trans_tuple=None, step =None):
        # go back to the program if we do not have sufficient transitions i.e. sufficient data in the replay buffer
        
        metrics = dict()
        
        if trans_tuple == None:
            # print('here')
            # print(self.memory.mem_cntr, self.batch_size)
            if self.memory.mem_cntr < self.batch_size:
                return

            # print("Collected enough data and have started learning...")
            # print("="*50)
            
            # to sample our buffer
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            # print("Original data")
            # for i in (state, action, reward, new_state, done): 
            #     print(i.shape)



        else: 

            self.use_tb = True
            
            state, action, reward, discount, new_state, step_type, next_step_type = trans_tuple

            # print("Original data for SLRL")

            

            # for i in trans_tuple: 
            #     print(i.shape)


            done = next_step_type.clone()
            done[done < 2] = 1
            done[done == 2] = 0

            if self.use_tb:
                metrics['batch_reward'] = reward.mean().item()

        # to transform numpy arrays to pytorch tensors
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)  # these are the actions sampled fro the replay buffer

        
        # to calculate the value current state and next state via the value and target value networks
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        # to set the terminal states value to be 0
        value_[done] = 0.0

        # to get the actions according to the new policy wihtout using the reparameterization trick
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)





        # to critic values under the new policy. NOTE: Using"actions" and not "action"
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)  # to overcome overestimation
        critic_value = critic_value.view(-1)

        # to define the VALUE NETWORK LOSS
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()


        
        # to define the ACTOR NETWORK loss
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_probs.mean().item()
            metrics['alpha_value'] = 1




        # to define the CRITIC NETWORK LOSS
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # to add the entropy term so as to encourage exploration
        q_hat = self.scale*reward.view(-1,) + self.gamma*value_

        # NOTE: using the action from the replay buffer
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # to update the target value network parameters
        self.update_network_parameters()
        
        if self.use_tb:
            metrics['critic_target_q'] = critic_1_loss.mean().item()
            metrics['critic_q1'] = q1_old_policy.mean().item()
            metrics['critic_q2'] = q2_old_policy.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        if trans_tuple !=None:
            return metrics
        




class CriticNetworkEmbed(nn.Module):

    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='/tmp/sac'):
         
        '''
        beta - it is the learning rate
        input dims - # input dimensions (obs)
        '''
        
        super(CriticNetworkEmbed, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.encoding = nn.Linear(7, 7)

        # directly passing the state and action pair together
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # output layer - outputting the value of Q(s,a) and so in a scalar value
        self.q = nn.Linear(self.fc2_dims, 1) 

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # to transfer everything to GPU
        self.to(self.device)

    
    def forward(self, state, action):
        # along the batch dimension
        task_encod = self.encoding(state[:,-7:])
        action_value = self.fc1(torch.cat([state[:,:-7], task_encod, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self):
          path = os.path.dirname(os.path.abspath(__file__)) + self.checkpoint_file + '.zip'
          self.load_state_dict(torch.load(path))



class ValueNetworkEmbed(nn.Module):

    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='/tmp/sac'):
        
        super(ValueNetworkEmbed, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.encoding = nn.Linear(7, 7)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)

        # outputting a scalar quantity
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    
    def forward(self, state):
        task_encod = self.encoding(state[:,-7:])
        state_value = self.fc1(torch.cat([state[:,:-7], task_encod], dim=1))
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self):
          path = os.path.dirname(os.path.abspath(__file__)) + self.checkpoint_file + '.zip'
          self.load_state_dict(torch.load(path))




class ActorNetworkEmbed(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=4, name='actor', chkpt_dir='/tmp/sac'):
        super(ActorNetworkEmbed, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        '''
        this is the scaling factor for the action space as we have tanh as the output. but the range of actions in the metaworld is btw
        -1 and 1 so we are not using this max_action variable or setting it to 1
        '''
        self.max_action = max_action

        # to avoid log or division by 0
        self.reparam_noise = 1e-6

        self.encoding = nn.Linear(7, 7)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # mean of the distribution for the policy; equal to the number of actions that we can take
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        # to define the standard deviation
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    
    def forward(self, state):
        task_encod = self.encoding(state[:, -7:])
        prob = self.fc1(torch.cat([state[:, :-7], task_encod], dim=1))
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        # to predict the mean and std using the same inpui.e. output of fc2
        mu = self.mu(prob) 
        sigma = self.sigma(prob)

        # to prevent the distribution from getting too big
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        # to sample the actions from the distribution
        if reparameterize:
            # if we want to add more noise into the sampled action
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        #action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        # this is to take the action in the env 
        action = torch.tanh(actions).to(self.device)

        # to calculate the loss wrt to the sampled action
        log_probs = probabilities.log_prob(actions)

        # to avoid dividing by 0
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self):
          path = os.path.dirname(os.path.abspath(__file__)) + self.checkpoint_file + '.zip'
          self.load_state_dict(torch.load(path))







