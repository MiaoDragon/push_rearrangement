import numpy as np
import matplotlib.pyplot as plt


def analyze_truncated_gauss_data():
    data = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/result.txt"

    a = np.loadtxt(data)

    print(a)

    plt.hist(a, bins=100, density=True)
    plt.show()


def analyze_spline_data():
    ts = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/spline_ts.txt"
    data = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/spline_data.txt"
    spline_order = 0
    ts = np.loadtxt(ts)
    data = np.loadtxt(data)

    # plot the data
    start_t = 0.0
    end_t = 7.0
    for spline_order in range(3):    
        for i in range(len(data[0])):
            plt.figure()
            plt.scatter(ts, data[:,i], c="red")
            samples = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/spline_sample_order_%d.txt"%(spline_order)
            samples = np.loadtxt(samples)
            sampled_t = np.linspace(start_t, end_t, len(samples)+1)[:-1]
            plt.plot(sampled_t, samples[:,i])
            plt.show()


analyze_spline_data()