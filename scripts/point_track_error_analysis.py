import numpy as np
import matplotlib.pyplot as plt


folder = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/"
robot_pos = np.loadtxt(folder+"robot_position.txt")
target_pos = np.loadtxt(folder+"target_position.txt")


plt.plot(robot_pos[:,0], label='robot')
plt.plot(target_pos[:,0], label='target')
plt.legend()
plt.show()
plt.plot(robot_pos[:,1], label='robot')
plt.plot(target_pos[:,1], label='target')
plt.legend()
plt.show()
plt.plot(robot_pos[:,2], label='robot')
plt.plot(target_pos[:,2], label='target')
plt.legend()
plt.show()
