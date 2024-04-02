"""
analyze how the policy changes
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_cost():
    directory = '/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/'
    f = 'policy_cost.txt'
    data = np.loadtxt(directory+f)
    for i in range(len(data[0])):
        plt.scatter(x=np.linspace(0,len(data)-1,len(data)), y=data[:,i])
    plt.show()

plot_cost()