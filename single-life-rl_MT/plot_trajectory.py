import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle 
# Create a numpy array with the list of 3D coordinates
env_names = ['sawyer_pick_place',       #0
            'sawyer_window_open',      #1
            'sawyer_window_close',     #2
            'sawyer_drawer_open',       #3
            'sawyer_drawer_close',      #4
            'sawyer_button_press',      #5
            'sawyer_push']              #6
env_idx = 1

title = " ".join([x.capitalize() for x in env_names[env_idx].split('_')])

paths= {}


# How to set paths better to automate this thing

paths[env_names[1]] = '/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/sawyer_window_open/2023.06.05.22.21.28_more_novely_far_away/discrimTrue_qwtTrue_seed42/trajectory_sawyer_window_open.pkl'


paths[env_names[2]] = '/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/sawyer_window_close/window_close_success/discrimTrue_qwtTrue_seed425/trajectory_sawyer_window_close.pkl'


paths[env_names[3]] = '/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/sawyer_drawer_open/2023.06.06.01.00.59/discrimTrue_qwtTrue_seed42/trajectory_sawyer_drawer_open.pkl'

paths[env_names[4]] = '/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/sawyer_drawer_close/2023.06.06.01.22.03/discrimTrue_qwtTrue_seed42/trajectory_sawyer_drawer_close.pkl'

paths[env_names[5]] = "/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/sawyer_button_press/2023.06.06.08.58.34/discrimTrue_qwtTrue_seed425/trajectory_sawyer_button_press.pkl"

path = paths[env_names[env_idx]]

object_positions_orig=[[0.2, 0.7 , 0.02],
                  [-0.04, 0.705, 0.16],
                  [0.3, 0.705, 0.16],
                  [0.0, 0.9, 0.0],
                  [0.0, 0.6, 0.02],
                  [0., 0.9, 0.0],
                  [0., 0.6, 0.02]]


goals_orig= [[0.5, 0.8, 0.02],
         [0.1, 0.69, 0.16],
         [0.14, 0.69 , 0.02],
         [0.0, 0.74, 0.05],
         [0.0, 0.885, 0.16],
         [0, 0.6 , 0.02],
         [0.1, 0.8, 0.02],
        ]

#Kept in order
object_positions=[[0.2, 0.7 , 0.02],
                  [0.1, 0.805, 0.16],
                  [0.36, 0.805, 0.16],
                  [0.125, 0.9, 0.0],
                  [0.25, 0.605, 0.16],
                  [0., 0.9, 0.0],
                  [0, 0.6 , 0.02]]

# Kept in order
goals = [[0.5, 0.8, 0.02],
         [0.2, 0.79, 0.16],
         [0.2, 0.79 , 0.02],
         [0.125, 0.74, 0.05],
         [0.25, 0.885 , 0.02],
         [0, 0.6 , 0.02],
         [0, 0.6 , 0.02],
        ]


starting_pos = [[0, 0.6, 0.2],
                [0, 0.4, 0.2],
                [0, 0.4, 0.2],
                [0, 0.6, 0.2],
                [0, 0.6, 0.2],
                [0, 0.6, 0.2],
                [0, 0.6, 0.2]]



env_name =env_names[env_idx]
object_name = env_name.split("_")[1].capitalize()
if object_name == 'Window':
    object_name += ' Arm'


f = open(path,'rb')

trajectory = pickle.load(f)

print(trajectory)
print(trajectory.shape, type(trajectory))

print('The length for the completion is ',trajectory.shape[0])


# Extract the x, y, and z coordinates from the trajectory array
x = trajectory[:, 0]
y = trajectory[:, 1]
z = trajectory[:, 2]

start = starting_pos[env_idx]
#Novel Positions
object_ =  object_positions[env_idx]
goal = goals[env_idx]

# Original Positions
object_orig =  object_positions_orig[env_idx]
goal_orig = goals_orig[env_idx]


# Create a 3D plot
fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
# ax.set_axis_off()
# Plot the trajectory

ax.scatter(x, y, marker='o', s = 1 , color = 'deepskyblue', alpha= 0.7)


if object_:
    ax.scatter(*object_[0:2], marker='o', color = 'red', s = 150 ,  label=f'{object_name} Position')

ax.scatter(*goal[0:2], marker='*', color = 'green' , s = 150,  label='Goal')


if object_:
    ax.scatter(*object_orig[0:2], marker="o", color = 'orangered', s = 150, alpha=0.3,label=f'Original {object_name} Position' )

ax.scatter(*goal_orig[0:2], marker='*', color = 'palegreen' , s = 150, label= 'Original Goal' )



ax.scatter(*start[0:2], marker='^', s = 100 , color = 'orange', alpha= 0.7, label = 'Start')


ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.1 ), ncol=5)

# Set labels for the axes with fontsize
ax.set_xlabel('X Coordinate of End Effector', fontsize=12)
ax.set_ylabel('Y Coordinate of End Effector', fontsize=12)
# ax.set_zlabel('Z', fontsize=12)
ax.set_xlim(-0.5, 0.5)

ax.set_ylim(0.3, 1)
# Set a title for the plot with fontsize
ax.set_title(f'Robotic Arm Trajectory: {title}', fontsize=14)

# Set fontsize for tick labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Show the plot
# plt.legend()
plt.savefig(f'results/{env_name}_trajectory.png')

plt.show()

# plt.savefig(f'results/{env_name}_trajectory.png')



