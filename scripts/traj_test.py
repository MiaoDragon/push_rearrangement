import numpy as np
import matplotlib.pyplot as plt

def test_sinusoid():
    start = np.loadtxt("/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/sinusoid_traj_test_data_start.txt")
    x_vec = np.loadtxt("/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/sinusoid_traj_test_data_x.txt")
    y_vec = np.loadtxt("/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/sinusoid_traj_test_data_y.txt")
    z_vec = np.loadtxt("/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/sinusoid_traj_test_data_z.txt")
    data = np.loadtxt("/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/sinusoid_traj_test_data.txt")

    x_data = (data - start).dot(x_vec)
    y_data = (data - start).dot(y_vec)
    z_data = (data - start).dot(z_vec)

    plt.plot(x_data, y_data)
    plt.show()
    plt.plot(z_data)
    plt.show()


test_sinusoid()