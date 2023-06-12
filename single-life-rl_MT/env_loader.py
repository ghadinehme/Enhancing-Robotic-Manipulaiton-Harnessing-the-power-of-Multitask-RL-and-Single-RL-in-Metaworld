# import envs.earl_benchmarks as earl_benchmark
# import envs.tabletop_manipulation as tabletop_manipulation
# from envs.half_cheetah_short_hurdle import HalfCheetahEnvShortHurdle
# from envs.half_cheetah_short import HalfCheetahEnvShort
# from envs.pointmass import PointMassEnv

from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_open_v2 import SawyerWindowOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_close_v2 import SawyerWindowCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_v2 import SawyerDrawerOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_close_v2 import SawyerDrawerCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_button_press_v2 import SawyerButtonPressEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import SawyerPushEnvV2
import numpy as np
import h5py
import json



from backend.wrappers import (
    ActionRepeatWrapper,
    ActionDTypeWrapper,
    ExtendedTimeStepWrapper,
    ActionScaleWrapper,
    DMEnvFromGymWrapper,
)

envs = [SawyerPickPlaceEnvV2, 
            SawyerWindowOpenEnvV2, 
            SawyerWindowCloseEnvV2,
            SawyerDrawerOpenEnvV2,
            SawyerDrawerCloseEnvV2,
            SawyerButtonPressEnvV2, SawyerPushEnvV2]

env_names= ['sawyer_pick_place', 'sawyer_window_open', 'sawyer_window_close', 'sawyer_drawer_open', 'sawyer_drawer_close','sawyer_button_press', 'sawyer_push']

# import the learned embeddings here of load the model here
d = [[np.sin(k*p) for k in range(1,len(envs)+1)] for p in range(1,len(envs)+1)]

env_to_task = {}
name_to_env = {}
for i in range(len(env_names)):
    env_to_task[env_names[i]] = np.array(d[i])
    name_to_env[env_names[i]] = envs[i]


def expert_dataloader(path=None):
     if path == None: 
          return None
     
     with open(path, 'rb') as f:
            
            temp = json.load(f)
            limit = 100000
            
            forward_demos = {}
            forward_demos['observation'] = np.array(temp['self.state_memory'][-limit:])
            forward_demos['action'] = np.array(temp['self.action_memory'][-limit:]).astype(np.float32) 
            forward_demos['reward'] = np.array(temp['self.reward_memory'][-limit:])
            forward_demos['terminal'] = np.array(temp['self.terminal_memory'][-limit:])[np.newaxis, :]
            forward_demos['next_observation'] = np.array(temp['self.new_state_memory'][-limit:])
            

            task_id_array = np.repeat(np.array([[1,0]]), repeats = forward_demos['observation'].shape[0], axis = 0)

            # print(task_id_array, task_id_array.shape)
            # print(forward_demos['observation'].shape)
            forward_demos['observation'] = np.concatenate((forward_demos['observation'],task_id_array), axis = 1)
            forward_demos['next_observation'] = np.concatenate((forward_demos['next_observation'],task_id_array), axis = 1)
            
            # print(forward_demos['observation'].shape)
            # exit(0)            
            return forward_demos




def make(name, frame_stack, action_repeat, resets=False, orig_dir=None):
    
    
    print(name)
    train_env = name_to_env[name]()
    eval_env = name_to_env[name]()


    # if name == 'sawyer':
    #     train_env = SawyerPickPlaceEnvV2()
    #     eval_env = SawyerPickPlaceEnvV2()

    # if name =='open_window':
    #     train_env = SawyerWindowOpenEnvV2()
    #     eval_env = SawyerWindowOpenEnvV2()
    #     forward_demos = None

    train_env._partially_observable = True
    train_env._freeze_rand_vec = True
    train_env._set_task_called = True
    train_env.reset()
    train_env._freeze_rand_vec = True

    eval_env._partially_observable = False
    eval_env._freeze_rand_vec = False
    eval_env._set_task_called = True
    eval_env.reset()
    eval_env._freeze_rand_vec = True
    
    reset_states = train_env.reset_model()
    goal_states = train_env._get_pos_goal()

    reset_states, goal_states = None, None

    # forward_demos = expert_dataloader(path = f'{orig_dir}/data/demos/{name}/pick_place_replay_buffer.json')
    forward_demos = None
            # reset_states = env_loader.get_initial_states()
    # Add open window environment here    

    # else:
    #     if name == 'kitchen':
    #         env_loader = earl_benchmark.EARLEnvs(
    #         name,
    #         reward_type="dense",
    #         reset_train_env_at_goal=False,
    #         train_resets=resets,
    #         )
    #     else:
    #         env_loader = earl_benchmark.EARLEnvs(
    #             name,
    #             reward_type="sparse",
    #             reset_train_env_at_goal=False,
    #             train_resets=resets,
    #         )
    #     train_env, eval_env = env_loader.get_envs()
    #     reset_states = env_loader.get_initial_states()
    #     reset_state_shape = reset_states.shape[1:]
    #     goal_states = env_loader.get_goal_states()
    #     forward_demos = None

    # add wrappers
    minimum = -1.0
    maximum = +1.0
    train_env = DMEnvFromGymWrapper(train_env)
    train_env = ActionDTypeWrapper(train_env, np.float32)
    train_env = ActionRepeatWrapper(train_env, action_repeat)
    train_env = ActionScaleWrapper(train_env, minimum=minimum, maximum=maximum)
    train_env = ExtendedTimeStepWrapper(train_env) 

    eval_env = DMEnvFromGymWrapper(eval_env)
    eval_env = ActionDTypeWrapper(eval_env, np.float32)
    eval_env = ActionRepeatWrapper(eval_env, action_repeat)
    eval_env = ActionScaleWrapper(eval_env, minimum=minimum, maximum=maximum)
    eval_env = ExtendedTimeStepWrapper(eval_env)

    return train_env, eval_env, reset_states, goal_states, forward_demos