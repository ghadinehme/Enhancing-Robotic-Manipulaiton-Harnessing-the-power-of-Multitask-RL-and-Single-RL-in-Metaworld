from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
import sys, time
import numpy as np
import sys
import cv2
import pygame

env = SawyerPickPlaceEnvV2()
env._partially_observable = False
env._freeze_rand_vec = False
env._set_task_called = True
obs = env.reset()
print("Initial obs: ", obs.shape)
print(obs)
#sys.exit()
env._freeze_rand_vec = True
lock_action = False
random_action = False
obs = env.reset()
action = np.array([0.54223, 0.12233,0.234,-0.3434])

  
pygame.init()
  
win = pygame.display.set_mode((500, 500))
  
# set the pygame window name 
pygame.display.set_caption("Moving rectangle")
  
# object current co-ordinates 
x = 0
y = 0
  
# dimensions of the object 
width = 20
height = 20
  
# velocity / speed of movement
vel = 10
  
# Indicates pygame is running
run = True
  
# for i in range(10000):



#     action = env.action_space.sample()
#     # print(action)


#     action = [0, 0, i/10000,0]
#     time.sleep(0.1)

#     ob, reward, done, info = env.step(action)

#     output = env.render()




# infinite loop 
while run:
    # creates time delay of 10ms 
    # pygame.time.delay(100)
      
    # iterate over the list of Event objects  
    # that was returned by pygame.event.get() method.  
    for event in pygame.event.get():
          
        # if event object type is QUIT  
        # then quitting the pygame  
        # and program both.  
        if event.type == pygame.QUIT:
              
            # it will make exit the while loop 
            run = False
# stores keys pressed 
    event = pygame.event.wait()
    keys = pygame.key.get_pressed()
    


    if keys[pygame.K_j]:
        action[0] -=0.001


    if keys[pygame.K_l]:
        action[1] -=0.001


    if keys[pygame.K_i]:
        action[2] -=0.001


    if keys[pygame.K_k]:
        action[3] -=0.001


    # if left arrow key is pressed
    if keys[pygame.K_LEFT] and x>0:
        
        # decrement in x co-ordinate
        action[0]+=0.001
        x -= vel
        
    # if left arrow key is pressed
    if keys[pygame.K_RIGHT] and x<500-width:
        
        # increment in x co-ordinate
        x += vel
        action[1]+=0.001
        
    # if left arrow key is pressed   
    if keys[pygame.K_UP] and y>0:
        
        # decrement in y co-ordinate
        y -= vel
        action[2]+=0.001
        
    # if left arrow key is pressed   
    if keys[pygame.K_DOWN] and y<500-height:
        # increment in y co-ordinate
        y += vel
        action[3]+=0.001


    ob, reward, done, info = env.step(action)
    # time.sleep(0.1)
    output = env.render()

    print(action)
    # cv2.imshow("img", output)
    # cv2.waitKey(0)

    # completely fill the surface object  
    # with black colour  
    win.fill((0, 0, 0))
    
    # drawing object on screen which is rectangle here 
    # pygame.draw.rect(win, (255, 0, 0), (x, y, width, height))
    
    # it refreshes the window
    pygame.display.update() 

# closes the pygame window 
pygame.quit()



# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so