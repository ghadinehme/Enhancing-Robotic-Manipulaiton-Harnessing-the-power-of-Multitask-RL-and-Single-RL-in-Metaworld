
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

import time

envs = [SawyerPickPlaceEnvV2, 
            SawyerWindowOpenEnvV2, 
            SawyerWindowCloseEnvV2,
            SawyerDrawerOpenEnvV2,
            SawyerDrawerCloseEnvV2,
            SawyerButtonPressEnvV2, SawyerPushEnvV2]

env_names= ['sawyer_pick_place', 'sawyer_window_open', 'sawyer_window_close', 'sawyer_drawer_open', 'sawyer_drawer_close','sawyer_button_press', 'sawyer_push']



train_env =envs[2]()

train_env._partially_observable = True
train_env._freeze_rand_vec = True
train_env._set_task_called = True
train_env.reset()

# train_env._freeze_rand_vec = True


while True: 

    train_env.render()
    # time.sleep(10)