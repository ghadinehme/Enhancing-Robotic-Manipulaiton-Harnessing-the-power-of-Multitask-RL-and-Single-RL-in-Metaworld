from envs.pointmass import PointMassEnv
from envs.tabletop_manipulation import TabletopManipulation
from envs.half_cheetah_short_hurdle import HalfCheetahEnvShortHurdle
import random 
import mujoco_py
import time
import cv2
import numpy as np 

# env = PointMassEnv(use_simulator=True)
env = HalfCheetahEnvShortHurdle()
for i in range(1000):
    action = [random.random(), random.random(),random.random(),random.random(),random.random(),random.random()]
    # action = [random.random(), random.random()]
    
    env.step(action)
    # time.sleep(0.1)
    data = env.render(mode ='human')
    # env.render()


    
    # print(data)
    # cv2.imshow("img", data)
    # cv2.waitKey(0)
    
    
    # if cv2.waitKey(1) == ord('q'):
    #     cv2.destroyAllWindows()
    
    # exit()