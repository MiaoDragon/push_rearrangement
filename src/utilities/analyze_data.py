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

def analyze_policy_sample():
    data = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/policy_sample_truncated.txt"
    N = 100
    knot_num = 10
    data = np.loadtxt(data)
    mat = np.zeros((N,knot_num,2))
    mat = data.reshape((N,knot_num,2))

    print('mat: ')
    print(mat)
    # print the first 10 of each knot
    for i in range(knot_num):
        print('first 10 of knot_num: ', i)
        print(mat[:10,i,:])


    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink']

    # for each knot, use a different color
    for i in range(knot_num):
        plt.scatter(mat[:,i,0],mat[:,i,1], c=colors[i])
    plt.show()


def analyze_policy_action():
    directory = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/"
    ts = directory + "policy_ts.txt"
    ts = np.loadtxt(ts)

    theta = directory + "policy_theta.txt"
    theta = np.loadtxt(theta)

    sampled_action = directory + "policy_sampled_actions.txt"
    sampled_action = np.loadtxt(sampled_action)

    sample_t0 = -0.5
    sample_tN = 10*0.1 + 0.5
    # for each control dimension, generate a plot of the action
    sampled_ts = np.linspace(sample_t0, sample_tN, len(sampled_action)+1)[:-1]

    for ctrl_idx in range(len(theta[0])):
        plt.figure()
        plt.scatter(ts, theta[:,ctrl_idx], c='red')
        plt.plot(sampled_ts, sampled_action[:,ctrl_idx])
        plt.show()

def analyze_clamped_policy_action():
    directory = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/"
    ts = directory + "clamp_policy_ts.txt"
    ts = np.loadtxt(ts)

    theta = directory + "clamp_policy_theta.txt"
    theta = np.loadtxt(theta)

    sampled_action = directory + "clamp_policy_sampled_actions.txt"
    sampled_action = np.loadtxt(sampled_action)

    sample_t0 = -0.5
    sample_tN = 10*0.1 + 0.5
    # for each control dimension, generate a plot of the action
    sampled_ts = np.linspace(sample_t0, sample_tN, len(sampled_action)+1)[:-1]

    for ctrl_idx in range(len(theta[0])):
        plt.figure()
        plt.scatter(ts, theta[:,ctrl_idx], c='red')
        plt.plot(sampled_ts, sampled_action[:,ctrl_idx])
        plt.show()

def analyze_shift():
    directory = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/"
    ts = directory + "shift_policy_ts.txt"
    ts = np.loadtxt(ts)

    theta = directory + "shift_policy_theta.txt"
    theta = np.loadtxt(theta)

    sampled_action = directory + "shift_policy_sampled_actions.txt"
    sampled_action = np.loadtxt(sampled_action)

    sample_t0 = -0.5
    sample_tN = 10*0.1 + 0.5
    # for each control dimension, generate a plot of the action
    sampled_ts = np.linspace(sample_t0, sample_tN, len(sampled_action)+1)[:-1]

    for ctrl_idx in range(len(theta[0])):
        plt.figure()
        plt.scatter(ts, theta[:,ctrl_idx], c='red')
        plt.plot(sampled_ts, sampled_action[:,ctrl_idx])
        plt.show()


def analyze_optmize():
    directory = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/build/"
    thetas = directory + "optimize_thetas.txt"
    thetas = np.loadtxt(thetas)
    N = 10
    n_knots = 5
    thetas = thetas.reshape((N,n_knots,2))
    Js = directory + "optimize_Js.txt"
    Js = np.loadtxt(Js)
    beta = 1.0
    rho = Js.min()
    ws = np.exp(-1/beta*(Js-rho))

    sum_weighted_theta = np.zeros((n_knots, 2))

    for i in range(N):
        sum_weighted_theta += ws[i] * thetas[i]
    sum_weighted_theta = sum_weighted_theta / ws.sum()

    print("result: ")
    print(sum_weighted_theta)

# analyze_spline_data()
# analyze_policy_sample()
# analyze_policy_action()
# analyze_clamped_policy_action()
# analyze_shift()
analyze_optmize()