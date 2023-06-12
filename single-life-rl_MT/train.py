import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
import sys
from pathlib import Path
import env_loader
import hydra
import numpy as np
import torch
import random
import utils
from PIL import Image
import time
import utils 
from dm_env import specs
from logger import Logger
from simple_replay_buffer import SimpleReplayBuffer
from video import TrainVideoRecorder
from agents import SACAgent, Discriminator
from backend.timestep import ExtendedTimeStep
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})

from SAC import Agent, AgentEmbed

import pickle
torch.backends.cudnn.benchmark = True


env_names = ['sawyer_pick_place', 
             'sawyer_window_open', 
             'sawyer_window_close', 
             'sawyer_drawer_open', 
             'sawyer_drawer_close',
             'sawyer_button_press', 
             'sawyer_push'
            ]

# Sine Cosine Embeddings
d = [[np.sin(k*p) for k in range(1,len(env_names)+1)] for p in range(1,len(env_names)+1)]

# One Hot Encoding for the tasks
# d = [[int(i == j) for i in range(len(env_names))] for j in range(len(env_names))]

env_to_task = {}

for i in range(len(env_names)):
    env_to_task[env_names[i]] = d[i]

# task = "sawyer_open_window"
# Do not use this
# env_to_task['sawyer_pick_place'] = np.array([0,1])
# env_to_task['sawyer_open_window'] = np.array([1,0])

# TODO: Make agent to SAC 
def make_agent(obs_spec, action_spec, cfg, env_name):
    
   
    
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.feature_dim = obs_spec.shape[0]
    return SACAgent(obs_shape=cfg.obs_shape,
                action_shape=cfg.action_shape,
                device=cfg.device,
                lr=cfg.lr,
                feature_dim=cfg.feature_dim,
                hidden_dim=cfg.hidden_dim,
                critic_target_tau=cfg.critic_target_tau, 
                reward_scale_factor=cfg.reward_scale_factor,
                use_tb=cfg.use_tb,
                from_vision=cfg.from_vision,
                env_name=env_name)
    
def make_discriminator(obs_spec, action_spec, cfg, env_name, discrim_type, mixup, q_weights, num_discrims):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.feature_dim = obs_spec.shape[0]
    return Discriminator(
            discrim_hidden_size=cfg.discrim_hidden_size,
            obs_shape=cfg.obs_shape,
            action_shape=cfg.action_shape,
            device=cfg.device,
            lr=cfg.lr,
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            critic_target_tau=cfg.critic_target_tau, 
            reward_scale_factor=cfg.reward_scale_factor,
            use_tb=cfg.use_tb,
            from_vision=cfg.from_vision,
            env_name=env_name,
            discrim_type=discrim_type,
            mixup=mixup,
            q_weights=q_weights,
            num_discrims=num_discrims,)

class Workspace:
    def __init__(self, cfg, orig_dir):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.orig_dir = orig_dir

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)


        self.obs_spec = specs.Array((46,), np.float32, 'observation')
        
        self.setup()

        
        if cfg.task_encoding == 'trainable':
            self.agent = AgentEmbed(env = self.train_env)
        else:
            self.agent = Agent(env = self.train_env)
        # Change this to observation + taskid
        # Discriminator type is state or state-action pair
        if self.cfg.use_discrim:
            self.discriminator = make_discriminator(self.obs_spec,
                                                  self.train_env.action_spec(),
                                                  self.cfg.agent,
                                                  self.cfg.env_name,
                                                     discrim_type=self.cfg.discrim_type,
                                                     mixup=self.cfg.mixup,
                                                     q_weights=self.cfg.q_weights,
                                                      num_discrims=self.cfg.num_discrims)
        
        self.timer = utils.Timer()
        self.timer._start_time = time.time()
        self._global_step = -self.cfg.num_pretraining_steps
        print("Global step", self._global_step)
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.train_env, self.eval_env, self.reset_states, self.goal_states, self.forward_demos = env_loader.make(self.cfg.env_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.resets, orig_dir=self.orig_dir)
        if self.cfg.resets:
            _, self.train_env, self.reset_states, self.goal_states, self.forward_demos = env_loader.make(self.cfg.env_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.resets, orig_dir=self.orig_dir)
    
        # Observation space plus taskid
        data_specs = ( self.obs_spec,
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
    
        self.replay_storage_f = SimpleReplayBuffer(data_specs,
                                                       self.cfg.replay_buffer_size,
                                                       self.cfg.batch_size,
                                                       self.work_dir / 'forward_buffer',
                                                  self.cfg.discount)
    
        self.online_buffer = SimpleReplayBuffer(data_specs,
                                                       self.cfg.replay_buffer_size,
                                                       self.cfg.batch_size,
                                                       self.work_dir / 'forward_buffer',
                                               self.cfg.discount,
                                               time_step=self.cfg.time_step)
        self.prior_buffers = []
        
        # We have a single discriminator for now
        for _ in range(self.cfg.num_discrims):
            self.prior_buffers.append(SimpleReplayBuffer(data_specs,
                                                       self.cfg.prior_buffer_size, 
                                                       self.cfg.batch_size,
                                                       self.work_dir / 'forward_buffer',
                                                         self.cfg.discount,
                                                         time_step=self.cfg.time_step,
                                                       q_weights=self.cfg.q_weights,
                                                       rl_pretraining=self.cfg.rl_pretraining))
        
        self._forward_iter = None 

        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None, self.cfg.env_name)


    @property
    def forward_iter(self):
        if self._forward_iter is None:
            self._forward_iter = iter(self.replay_storage_f)
        return self._forward_iter
    
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat
    
    def save_im(self, im, name):
        img = Image.fromarray(im.astype(np.uint8)) 
        img.save(name)

    def save_gif(self, ims, name):
        imageio.mimsave(name, ims, fps=len(ims)/100)
    
        

    def train(self, snapshot_dir=None):
        # predicates
        train_until_step = utils.Until(self.cfg.online_steps, 
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_init_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        if self.cfg.rl_pretraining:
            time_step = self.eval_env.reset()
            _, self.eval_env_pretraining, _, _, _ = env_loader.make(self.cfg.env_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.resets, orig_dir=self.orig_dir)
        else:
            # TODO: reset error here
            self.train_env._set_task_called = True
            time_step = self.train_env.reset()
        dummy_action = time_step.action

        if self.forward_demos and (not self.cfg.rl_pretraining or self.cfg.use_demos):
            # This adds offline_data from the environment setup itself.
            # Adds the offline_data to the replay_storage and the prior_buffer
            self.replay_storage_f.add_offline_data(self.forward_demos, dummy_action, env_name=self.cfg.env_name)
            
            for buffer in self.prior_buffers:
                _ = buffer.add_offline_data(self.forward_demos, dummy_action, env_name=self.cfg.env_name)
        
        # Iterator for the prior buffer
        prior_iters = []
        
        for d in range(self.cfg.num_discrims):
            prior_iters.append(iter(self.prior_buffers[d])) 
        
        # print('WHAT IS THIS PRIOR')
        # print(prior_iters)


        # Populated during the online run
        online_iter = iter(self.online_buffer)
        # This is the agent and the buffer
        cur_agent = self.agent
        cur_buffer = self.replay_storage_f
        
        # What is this used ?
        cur_iter = self.forward_iter

        if self.cfg.save_train_video:
            self.train_video_recorder.init(self.train_env)
    
        metrics = None
        episode_step, episode_reward = 0, 0
        # distances = []
        # num_stuck = 0
        past_timesteps = []
        online_rews = [] # all online rewards
        online_qvals = [] # all online qvals
        end_effector = [] # record the arm position


        logs = {}        


        cur_reward = torch.tensor(0.0).cuda()
        counter = 0
        # initial_back = 0. # For cheetah
        while train_until_step(self.global_step):
            
            '''Start single episode'''
            if self.global_step == 0:
                print("Starting single episode")
                time_step = self.train_env.reset()


                logs['starting_gripper_pos'] = time_step.observation[0:3]
                logs['starting_object_pos'] = time_step.observation[4:7]
                logs['goal_position'] = time_step.observation[-3:]
            

                logs['observation'] = []
                # print(time_step.observation.shape, np.array([1,0]).reshape((-1,)).shape )
                time_step =  ExtendedTimeStep(
                                                observation=np.concatenate( (time_step.observation, env_to_task[self.cfg.env_name]), axis = 0 ),       step_type=time_step.step_type,
                                                action=time_step.action,
                                                reward=time_step.reward,
                                                discount=time_step.discount
                                            )
                
                # print(time_step.observation.shape)
                cur_buffer.add(time_step)
                # exit()
                # x_progress = []
                # y_progress = []
                # agent_x = []
                # agent_y = []
                
                if self.cfg.rl_pretraining or True:
                    # Important step here
                    # print('Here')
                    min_q, max_q = cur_buffer.load_buffer(f'{snapshot_dir}/', 
                                                          self.cfg.prior_buffer_size, 
                                                          self.cfg.env_name, 
                                                          agent = self.agent,
                                                          taskid = env_to_task[self.cfg.env_name])
                    # print('Q values')
                    # print(min_q, max_q)
                    if self.prior_buffers[0].__len__() == 0:
                        _, _ = self.prior_buffers[0].load_buffer(f'{snapshot_dir}/', self.cfg.prior_buffer_size, agent = self.agent, taskid=env_to_task[self.cfg.env_name])
                    
            '''Logging and eval'''
            criteria = self.global_step % 500 == 0 
            # if self.cfg.resets and time_step.last():
            #     episode_step, episode_reward = 0, 0
            #     time_step = self.train_env.reset()
            #     cur_buffer.add(time_step)
            if criteria: 
                if self.global_step == 0:
                    episode_step, episode_reward = 0, 0
                self._global_episode += 1
                


                with open(str(self.work_dir / 'trajectory_') + self.cfg.env_name + '.pkl', 'wb') as f:
                    
                    logs['observations'] = end_effector
                    pickle.dump(logs, f) 
                    
                    # pickle.dump(np.array(end_effector), f) 

                # if self.global_step % 1000 == 0 and self.global_step > 0:
                    # if self.cfg.save_train_video:
                        # eval('export LD_PRELOAD=')
                        # print('Saving Video Frame and GIF')
                        # self.save_im(self.train_env.render(mode = 'rgb_array',offscreen = False), f'{self.work_dir}/train_video/train{self.global_frame}.png')
                        # self.train_video_recorder.save(f'train{self.global_frame}.gif')
                        # eval('export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('forward_buffer_size', len(self.replay_storage_f))
                        log('step', self.global_step)
                
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                
             ##############################################################################################  
            '''If online during single episode'''
            if self.global_step >= 0 or self.cfg.rl_pretraining:
                '''Sample action'''
                
                # Critic is forzen and only act here
                # print('Rendering Now')
                # while True:    
                
                with torch.no_grad():
                        # obs = np.append(time_step.observation, np.array([1,0])).astype("float32")
                        obs=time_step.observation
                        # print(obs)
                        action = cur_agent.choose_action(obs)
                        # print(action)
                        # Evaluation mode should use the mean action while training should use sample action

                # Now the end_effector position is stored
                end_effector.append(obs[0:3])
                
                # print('Between this')
                '''Take env step'''                           
                if self.cfg.rl_pretraining and self.global_step < 0:
                    time_step = self.eval_env.step(action)
                    self.eval_env.render()
                else:
                    
                    time_step = self.train_env.step(action)

                    self.train_env.render()
                # time.sleep(0.1)
                
                # print(counter)

                # if counter > 500:
                #     exit()
                # counter +=1
                #This time_step has dimension 39
                # print(time_step.observation.shape)
                # time_step.observation = np.append(time_step.observation, np.array([1,0]))
                time_step =  ExtendedTimeStep(
                                                observation=np.concatenate( (time_step.observation, env_to_task[self.cfg.env_name]), axis = 0),       step_type=time_step.step_type,
                                                action=time_step.action,
                                                reward=time_step.reward,
                                                discount=time_step.discount
                                            )
                


                orig_reward = time_step.reward

                online_rews.append(cur_reward.detach().item())
                with torch.no_grad():
                    Q1 = self.agent.critic_2(torch.FloatTensor(time_step.observation).view(1,-1).cuda(), torch.FloatTensor(time_step.action).view(1,-1).cuda())
                    # obs = np.append(time_step.observation, np.array([1,0]))
                    # Q1= self.agent.target_value(torch.FloatTensor(obs).cuda())
                
                online_qvals.append(Q1.detach().item())
               
                success_criteria = False
               
                success_criteria = (time_step.step_type == 2)
                

                if success_criteria or self.global_step == self.cfg.online_steps - 1:
                    
                    time_step = ExtendedTimeStep(observation=time_step.observation,
                                                 step_type=2,
                                                 action=action,
                                                 reward=time_step.reward,
                                                 discount=time_step.discount)
                    print("Completed task in steps", self.global_step, time_step)
                   
                    with open(str(self.work_dir / 'trajectory_') + self.cfg.env_name + '.pkl', 'wb') as f:
                        
                        
                        logs['observations'] = end_effector
                        # pickle.dump(np.array(end_effector), f) 
                        pickle.dump(logs, f) 





                    with open(f"{self.work_dir}/total_steps.txt", 'w') as f:
                        f.write(str(self.global_step))
                    if self.cfg.save_train_video:
                        self.train_video_recorder.save(f'train{self.global_step}.mp4')
               
                    print('Total Steps', len(end_effector))
                    exit()
                    
                episode_reward += orig_reward
                if self.cfg.biased_update and self.global_step % self.cfg.biased_update == 0:
                    # Use a biased TD update to control critic values
                    # np.append( time_step.observation, np.array([0,1]))
                    time_step = ExtendedTimeStep(observation = time_step.observation,
                                             step_type=2,
                                             action=action,
                                             reward=orig_reward,
                                             discount=time_step.discount)
                
                # Add to buffer
                if self.cfg.rl_pretraining and self.global_step < 0:
                    self.prior_buffers[0].add(time_step)
                else:
                    cur_buffer.add(time_step)
                    self.online_buffer.add(time_step)
                episode_step += 1
                
            if self.cfg.save_train_video and self.global_step < 50000:
                self.train_video_recorder.record(self.train_env)

           ##############################################################################################    
            if self.cfg.use_discrim:
                if self.global_step % self.cfg.discriminator.train_interval == 0 and self.online_buffer.__len__() > self.cfg.discriminator.batch_size:
                    for k in range(self.cfg.discriminator.train_steps_per_iteration):
                        if self.cfg.rl_pretraining and self.cfg.q_weights:
                       
                            metrics = self.discriminator.update_discriminators(pos_replay_iter = prior_iters, 
                                                                               neg_replay_iter = online_iter, 
                                                                               val_function = self.agent.critic_1, 
                                                                               current_val = cur_reward, 
                                                                               current_obs = time_step, 
                                                                               min_q = min_q, 
                                                                               max_q = max_q, 
                                                                               baseline=self.cfg.baseline,
                                                                               task_id = env_to_task[self.cfg.env_name])
                        else:
                            metrics = self.discriminator.update_discriminators(pos_replay_iter = prior_iters, 
                                                                               neg_replay_iter = online_iter, 
                                                                               val_function = self.agent.critic_1, 
                                                                               current_val = cur_reward, 
                                                                               current_obs = time_step, 
                                                                               min_q = min_q, 
                                                                               max_q = max_q, 
                                                                               task_id = env_to_task[self.cfg.env_name],
                                                                               baseline = 0.1)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if not seed_until_step(self.global_step):
                if self.cfg.use_discrim and self.online_buffer.__len__() > self.cfg.discriminator.batch_size:
                    trans_tuple, original_reward = self.discriminator.transition_tuple(cur_iter)
              
                    metrics = cur_agent.learn(trans_tuple, self.global_step)
                    
                    metrics['original_reward'] = original_reward.mean()

                    if len(past_timesteps) > 10: # for logging
                        del past_timesteps[0]
                    past_timesteps.append(time_step)
                    old_time_step = past_timesteps[0]
                    latest_tuple, original_reward = self.discriminator.transition_tuple(cur_iter, cur_time_step=time_step, old_time_step=old_time_step)
                    _, _, latest_reward = latest_tuple
                    actual_reward, disc_s = latest_reward
                    metrics['latest_r'] = actual_reward
                    metrics['disc_s'] = disc_s
                    cur_reward = disc_s # Use latest discriminator score as baseline val
                else:
                    obs, action, reward, discount, next_obs, step_type, next_step_type, idxs, q_vals  = utils.to_torch(next(cur_iter), 'cuda:0')
                    trans_tuple = (obs, action, reward, discount, next_obs, step_type, next_step_type)
                    
                    metrics = cur_agent.learn(trans_tuple, self.global_step)
                    # metrics = cur_agent.update(trans_tuple, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            
            self._global_step += 1    
           


        with open(str(self.work_dir / 'trajectory_') + self.cfg.env_name + '.pkl', 'wb') as f:
            pickle.dump(np.array(end_effector), f) 



        if self.cfg.save_train_video:
            self.train_video_recorder.save(f'train{self.global_step}.mp4')

                  
    def save_snapshot(self, epoch=None):
        snapshot = self.work_dir / 'snapshot.pt'
        if epoch: snapshot = self.work_dir / f'snapshot{epoch}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    # Loads the workspace for the single RL non episodic run
    def load_snapshot(self, dirname=None):
        if dirname: 
            payload = torch.load(dirname)
        else: 
            snapshot = self.work_dir / 'snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        self._global_step = -self.cfg.num_pretraining_steps

# Add the environment of Sawyer along with the original dataset
@hydra.main(config_path='./', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd
    print(cfg)
    orig_dir = hydra.utils.get_original_cwd()
    workspace = W(cfg, orig_dir)
    snapshot_dir = None

    path = "/tmp/sac/mtsac_replay_buffer.json"
    workspace.agent.memory.import_buffer([path])
    workspace.agent.load_models()
   
    workspace.train(snapshot_dir)
    # workspace.train(snapshot_dir)


if __name__ == '__main__':
    main()
    

    
