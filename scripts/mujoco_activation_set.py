"""
try to set the activation of the model and see what happens
"""

import mujoco
import mujoco_viewer
import glfw
import numpy as np
import transformations as tf

path = "/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/3d_pendulum.xml"

model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

# render = mujoco.Renderer(model, 480, 640)

# try setting the pendulum activation
print('number of activation: ')
print(model.na)



data.act[0] = 0
data.act[1] = 30 * np.pi / 180
data.act[2] = 0
# above does not jump to the target activation

# activation 1: rotation around x axis
# activation 2: rotation around y axis
# activation 3: rotation around z axis


# obtain the sensor, and compare with the activation
print(data.sensor("sensor_quat").data)

while True:
    mujoco.mj_step(model, data)

    qw, qx, qy, qz = data.sensor("sensor_quat").data
    R = tf.quaternion_matrix([qw,qx,qy,qz])
    angle, direct, point = tf.rotation_from_matrix(R)

    act1 = angle*direct.dot(np.array([1,0,0]))
    act2 = angle*direct.dot(np.array([0,1,0]))
    act3 = angle*direct.dot(np.array([0,0,1]))
    print('activation: ')
    print(act1, ', ', act2, ', ', act3)




    if viewer.is_alive:
        glfw.make_context_current(viewer.window)
        viewer.render()
