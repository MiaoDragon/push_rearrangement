"""
implement the control barrier function for moving the end-effector to a target pose.
objective:
min_u ||([J_p,J_r]^Tu-[K_d*e_p(t),1/T(t)log(e_r(t))_V])||
constraints:
- joint angle control barrier function
    h_lb[j] = q[j]-q_lb[j], h_ub[j] = q_ub[j]-q[j]
- collision avoidance control barrier function
    alpha = 1*L_{max}*||q_dot_max||/h_min
    dh/dp*dp/dq*u+alpha*h(q) >= 0
- self collision avoidance control barrier function
    between enabled collision pairs, use the same formulation as collision avoidance.
    (it seems that Mujoco automatically filter out parent-child collisions. So we can directly
     use the contact information in the data structure)
"""


import mujoco
import mujoco.viewer
import numpy as np
import time
import transformations as tf
import cvxpy as cvx

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 2

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 1
Kori: float = 1

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.02

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
# max_angvel = 0.785
max_angvel = 15*np.pi/180
# collision avoidance
min_col_dist = 0.15
gamma = 1.0

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("/home/yinglong/Documents/research/task_motion_planning/non-prehensile-manipulation/project/push_rearrangement/xmls/robot_ik.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_name = "left_ee_tip"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
                "torso_joint_b1",
                "arm_left_joint_1_s",
                "arm_left_joint_2_l",
                "arm_left_joint_3_e",
                "arm_left_joint_4_u",
                "arm_left_joint_5_r",
                "arm_left_joint_6_b",
                "arm_left_joint_7_t"
    ]

    lb = [ -1.58, -3.13, -1.9, -2.95, -2.36, -3.13, -1.9, -3.13 ]
    ub = [ 1.58, 3.13, 1.9, 2.95, 2.36, 3.13, 1.9, 3.13 ]


    dof_ids = np.array([model.joint(name).id for name in joint_names])
    qvel_ids = np.array([model.joint(name).dofadr[0] for name in joint_names])
    qpos_ids = np.array([model.joint(name).qposadr[0] for name in joint_names])
    # actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Mocap body we will control with our mouse.
    mocap_name = "ee_goal"
    mocap_id = model.body(mocap_name).mocapid[0]
    goal_id = model.body(mocap_name).id

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(3)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    qpos_init = [0, 1.75, 0.8, 0, -0.66, 0, 0, 0]
    for i in range(len(qpos_init)):
        data.qpos[qpos_ids[i]] = qpos_init[i]

    # get data
    robot_geoms = []
    for i in range(model.ngeom):
        rootid = model.body(model.geom(i).bodyid).rootid
        if model.body(rootid).name == "motoman_base":
            if model.body(model.geom(i).bodyid).name == "motoman_base":
                continue
            robot_geoms.append(i)

    # excluded pairs
    excluded_pairs = []
    for i in range(len(model.exclude_signature)):
        a = model.exclude_signature[i] >> 16;
        b = model.exclude_signature[i] & 0xFFFF
        print('exclude pair: ')
        print(model.body(a).name)
        print(model.body(b).name)
        excluded_pairs.append((a,b))

    # adding robotiq gripper
    exclude_links = ["arm_right_link_7_t", "base", "base_mount", "right_driver", "right_coupler", "right_spring_link", "right_follower", "right_pad"]
    exclude_links += ["left_driver", "left_coupler", "left_spring_link", "left_follower", "left_pad"]
    for i in range(len(exclude_links)):
        for j in range(i+1,len(exclude_links)):
            a = model.body(exclude_links[i]).id
            b = model.body(exclude_links[j]).id
            excluded_pairs.append((a,b))



    excluded_pairs = set(excluded_pairs)
    # generating collision pairs for the robot
    collision_pairs = []
    for i in range(len(robot_geoms)):
        for j in range(i+1,len(robot_geoms)):
            geom1 = robot_geoms[i]
            geom2 = robot_geoms[j]
            # if the two bodies are parent-child, skip
            if model.body(model.geom(geom1).bodyid).parentid == model.geom(geom2).bodyid:
                continue            
            if model.body(model.geom(geom2).bodyid).parentid == model.geom(geom1).bodyid:
                continue
            if model.body(model.geom(geom1).bodyid) == model.body(model.geom(geom2).bodyid):
                continue
            if (model.geom(geom1).bodyid[0],model.geom(geom2).bodyid[0]) in excluded_pairs:
                continue
            if (model.geom(geom2).bodyid[0],model.geom(geom1).bodyid[0]) in excluded_pairs:
                continue
            # should satisfy the compatibility check
            if (model.geom(geom1).contype & model.geom(geom2).conaffinity) or (model.geom(geom2).contype & model.geom(geom1).conaffinity):
                collision_pairs.append((robot_geoms[i],robot_geoms[j]))
    # for i in range(model.nbody):
    #     for j in range(i+1,model.nbody):
    #         rootid1 = model.body(i).rootid
    #         rootid2 = model.body(j).rootid
    #         if model.body(rootid1).name != "motoman_base":
    #             continue
    #         if model.body(rootid2).name != "motoman_base":
    #             continue
    #         # if the two bodies are parent-child, skip
    #         if model.body(i).parentid == j or model.body(j).parentid == i:
    #             continue
    #         # if the two bodies are in disabled collision group, skip
    #         collision_pairs.append((i,j))
    #         print('added collision pair:')
    #         print(model.body(i).name)
    #         print(model.body(j).name)


    viewer_iter = 0
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer_iter += 1
        # Reset the simulation.
        # mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()
            q = data.qpos.copy()  # Note the copy here is important.
            q_selected = q[qpos_ids]
            
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            u = cvx.Variable(len(qvel_ids))

            # * objective function *
            e_p = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            e_r = data.site(site_id).xmat.reshape((3,3))
            e_r = data.body(goal_id).xmat.reshape((3,3)) @ e_r.T
            mat = np.eye(4)
            mat[:3,:3] = e_r
            ang, direct, _ = tf.rotation_from_matrix(mat)
            e_r = ang*direct
            error = np.zeros((6))
            error[:3] = e_p
            error[3:] = e_r
            # error = error / np.linalg.norm(error) * 5.0*np.pi/180
            error = error / integration_dt

            jac = np.zeros((6, model.nv))
            mujoco.mj_jacBody(model, data, jac[:3], jac[3:], model.body("left_ee_tip").id)
            # select the subset of the Jacobian
            jac_selected = np.zeros((6,len(qvel_ids))).astype(float)
            jac_selected = jac[:,qvel_ids]
            # minimize: ||J*q_dot - error||_2
            #  minimize q_dot^T J^T J q_dot - 2 error^T J q_dot + error^T error
            P = jac_selected.T @ jac_selected
            q_vec = -2 * error @ jac_selected
            # alpha = 1 * 2 * np.linalg.norm(ub) / min_col_dist
            alpha = 0.1  # small alpha can have more successful collision avoidance
                         # but collision can still happen
            constrs = []
            print('##############################################')
            print('collision with shelf...')
            # * handle collision with the shelf *
            for i in range(len(robot_geoms)):
                from_to = np.zeros(6)
                dist = mujoco.mj_geomDistance(model, data, robot_geoms[i], model.geom("shelf_bottom").id,
                                    2*min_col_dist, from_to)
                if dist >= 2*min_col_dist:
                    continue
                # constraint: dh/dp*dp/dq*u+alpha*h(q) >= 0
                # where h(q) is the distance to the shelf

                # the fromto vector tells us the direction to move toward the collision
                # the jacobian of the collision should be dg/dx dx/dq
                # compute the jacobian w.r.t. the collsion point on the geom i
                body_id = model.geom(i).bodyid
                jac_collision = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jac_collision[:3], None, from_to[:3], body_id)
                dhdx = -(from_to[3:]-from_to[:3])
                # if dist < 0:
                #     dgdx = -dgdx
                dhdx = dhdx/np.linalg.norm(dhdx)#*dist
                jac_collision = jac_collision[:,qvel_ids]
                jac_collision = dhdx.dot(jac_collision)
                # dBdq = dB/dh dh/dx dx/dq
                # B(h) = 1/h
                # dB/dh = -1/h^2
                # constraints: dB/dq*u - gamma/B(h) <= 0
                # dBdh = -1/(dist**2)
                # Bh = 1/dist
                # constrs.append(dBdh*jac_collision@u - gamma/Bh <= 0)
                constrs.append(jac_collision @ u + alpha * dist >= 0)
                print('jac_collision: ')
                print(jac_collision)
                print('dist: ')
                print(dist)


            # * handle self collision *
            # h(q) = h(p1,p2)
            # dh/dq = dh/dp1 dp1/dq + dh/dp2 dp2/dq
            print('##############################################')
            print('self collision:')
            for i in range(len(collision_pairs)):
                from_to = np.zeros(6)
                geom1 = collision_pairs[i][0]
                geom2 = collision_pairs[i][1]
                dist = mujoco.mj_geomDistance(model, data, geom1, geom2,
                                    min_col_dist, from_to)
                if dist >= min_col_dist:
                    continue
                # print('geom pairs: ')
                # print(model.geom(geom1).name)
                # print(model.geom(geom2).name)
                # print('body names:')
                # print(model.body(model.geom(geom1).bodyid).name)
                # print(model.body(model.geom(geom2).bodyid).name)
                # print('distance: ', dist)
                body_id1 = model.geom(geom1).bodyid
                jac_collision1 = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jac_collision1[:3], None, from_to[:3], body_id1)
                jac_collision1 = jac_collision1[:,qvel_ids]
                dhdx1 = -(from_to[3:]-from_to[:3])
                dhdx1 = dhdx1/np.linalg.norm(dhdx1)
                body_id2 = model.geom(geom2).bodyid
                jac_collision2 = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jac_collision2[:3], None, from_to[3:], body_id2)
                jac_collision2 = jac_collision2[:,qvel_ids]
                dhdx2 = -(from_to[:3]-from_to[3:])
                dhdx2 = dhdx2/np.linalg.norm(dhdx2)
                dhdq = dhdx1.dot(jac_collision1) + dhdx2.dot(jac_collision2)
                constrs.append(dhdq @ u + alpha * dist >= 0)
                print('dhdq: ')
                print(dhdq)
                print('dist: ')
                print(dist)
            print('##############################################')


            # joint limit constraints
            # h(q) = q_ub - q
            # dh/dq = -I
            constrs.append(-u + 1.0*(np.array(ub)-q_selected)>=0)
            # h(q) = q - q_lb
            # dh/dq = I
            constrs.append(u + 1.0*(q_selected-np.array(lb))>=0)

            max_angvels = np.zeros((len(qvel_ids))) + max_angvel
            constrs.append(u >= -max_angvels)
            constrs.append(u <= max_angvels)

            print('error: ')
            print(error)

            prob = cvx.Problem(cvx.Minimize(cvx.quad_form(u, P) + q_vec @ u), constrs)
            # prob.solve(verbose=True)
            prob.solve(solver=cvx.ECOS)
            # Print result.
            print("status:", prob.status)
            print("The optimal value is", prob.value)
            print("A solution x is")
            print(u.value)
            print('J*u:')
            print(jac_selected @ u.value)
            # input('next...')            
            
            dq_total = np.zeros((model.nv))
            dq_total[qvel_ids] = u.value

            # Clamp maximum joint velocity.
            # dq_abs_max = np.abs(dq_total).max()
            # if dq_abs_max > max_angvel:
            #     dq_total *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq_total, dt)  # q should be of size nq
            # np.clip(q, *model.jnt_range.T, out=q)
            # select the ones to update
            q_selected = q[qpos_ids]
            np.clip(q_selected, lb, ub, out=q_selected)

            data.qpos[qpos_ids] = q_selected#q[qpos_ids]
            # Set the control signal and step the simulation.
            # data.ctrl[actuator_ids] = q[dof_ids]
            # mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            
            viewer.sync()
            # time_until_next_step = dt - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()