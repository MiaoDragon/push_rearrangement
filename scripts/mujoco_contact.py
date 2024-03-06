import mujoco


prob_path = '/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/mujoco_mpc/mjpc/tasks/robot_push/task.xml'
model = mujoco.MjModel.from_xml_path(prob_path)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

print('num of contacts: ', len(data.contact))


for i in range(len(data.contact)):
    print('contact ', i, '...')
    print(data.contact[i])
    # print(data.geom(data.contact[i].geom1))
    # print(data.geom(data.contact[i].geom2))

# print(data.body('arm_left_link_7_t'))

# print(model.body('arm_left_link_7_t'))
# print(model.body(model.body_rootid[model.body('object').id]))