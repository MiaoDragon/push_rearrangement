import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/robot_position_error.txt")

plt.figure()
plt.plot(data[:,0])
plt.show()