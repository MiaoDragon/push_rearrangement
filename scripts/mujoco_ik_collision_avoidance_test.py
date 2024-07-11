"""
This script demonstrates how to perform inverse kinematics with collision avoidance.
uses the mj_geomDistance function to check the distance between two geoms, and provide
the jacobian for the collision avoidance, used in null-space IK methods.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import transformations as tf

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

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
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785


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
        print('rootid: ')
        print(model.body(rootid).name)
        if model.body(rootid).name == "motoman_base":
            robot_geoms.append(i)



    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        # mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            # Spatial velocity (aka twist).
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # transform 1->2
            # transform1 = np.eye(4)            
            # transform1[:3,:3] = data.site(site_id).xmat.reshape((3,3))
            # transform1[:3,3] = data.site(site_id).xpos
            # transform2 = np.eye(4)
            # transform2[:3,:3] = data.body(goal_id).xmat.reshape((3,3))
            # transform2[:3,3] = data.body(goal_id).xpos
            # transform = transform2@np.linalg.inv(transform1)
            # # convert to twist
            # ang, direc, _ = tf.rotation_from_matrix(transform)
            # twist = np.zeros((6))  # position, rotaton
            # twist[3:] = ang * direc
            # twist[:3] = transform[:3,3]
            # twist = twist / integration_dt

            # Jacobian
            mujoco.mj_jacBody(model, data, jac[:3], jac[3:], model.body("left_ee_tip").id)

            # select the subset of the Jacobian
            jac_selected = np.zeros((6,len(qvel_ids))).astype(float)
            jac_selected = jac[:3,qvel_ids]
            # dq = np.linalg.pinv(jac_selected).dot(twist)
            dq = jac_selected.T @ np.linalg.solve(jac_selected @ jac_selected.T + diag, twist[:3])

            # add here extended Jacobian method
            # shelf_bottom
            # simple extended Jacobian: directly use the constraint/objective as additional constraints
            # (TODO) null-space extended Jacobian: project the constraint/objective to the nullspace of the Jacobian
            # this is more involved, and requires the gradient computation of the Jacobian
            # here we implement the simple extended Jacobian
            # * One method: use null-space formulation to maximize g(q), where g(q) is a funciton of the distance to collision
            gmin = 0.02  # threshold to keep
            threshold = 0.1

            dgdq = np.zeros(len(qvel_ids))

            # for visualization
            viewer.user_scn.ngeom = 0
            # evalute collisions for each robot geoms
            ngeom_i = 0
            for i in range(len(robot_geoms)):
                from_to = np.zeros(6)
                dist = mujoco.mj_geomDistance(model, data, i, model.geom("shelf_bottom").id,
                                    0.1, from_to)
                if dist >= threshold:
                    continue
                # rotation mat according to the frame
                rot_mat = np.eye(3)
                z_axis = from_to[3:]-from_to[:3]
                z_axis = z_axis/np.linalg.norm(z_axis)
                x_axis = np.array([1,0,0])
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis/np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                rot_mat[:3,0] = x_axis
                rot_mat[:3,1] = y_axis
                rot_mat[:3,2] = z_axis
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[ngeom_i],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.array([0.01,0.01,np.abs(dist)]),
                    pos=from_to[:3],
                    mat=rot_mat.flatten(),
                    rgba=np.array([1,0,0,0.5]))
                ngeom_i += 1
                
                # the fromto vector tells us the direction to move toward the collision
                # the jacobian of the collision should be dg/dx dx/dq
                # compute the jacobian w.r.t. the collsion point on the geom i
                body_id = model.geom(i).bodyid
                jac_collision = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jac_collision[:3], None, from_to[:3], body_id)
                dgdx = from_to[3:]-from_to[:3]
                if dist < 0:
                    dgdx = -dgdx
                dgdx = -dgdx/np.linalg.norm(dgdx)#*dist
                jac_collision = jac_collision[:,qvel_ids]
                jac_collision = dgdx.dot(jac_collision)
                dgdq += jac_collision
            viewer.user_scn.ngeom = ngeom_i
            dgdq = dgdq / len(robot_geoms)

            # * the constraints are: f(q)=x, g(q)>=g_min
            # hence when g(q)>=g_min, we can keep g(q+dq)=g(q), or just let it be free
            # when g(q)<g_min, we can set g(q+dq)=g_min



            # # Damped least squares.
            # dq_ = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

            # dq = dq_total


            # dq = dq_

            # Nullspace control biasing joint velocities towards the home configuration.
            # dq += (eye - np.linalg.pinv(jac) @ jac) @ (Kn * (q0 - data.qpos[dof_ids]))
            print('dgdq: ')
            print(dgdq)
            dq_null = (np.eye(jac_selected.shape[1]) - np.linalg.pinv(jac_selected) @ jac_selected) @ (dgdq)
            dq_null = dq_null / integration_dt
            print('dq_null: ')
            print(dq_null)
            # dq = dq + 10*dgdq
            dq = dq + dq_null
            
            dq_total = np.zeros((model.nv))
            dq_total[qvel_ids] = dq


            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq_total).max()
            if dq_abs_max > max_angvel:
                dq_total *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq_total, integration_dt)  # q should be of size nq
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
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()