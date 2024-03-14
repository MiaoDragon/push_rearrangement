/**
 * construct the "contact" data structure.
 */

#pragma once
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>

#include "../utilities/utilities.h"


enum BodyType {ROBOT, OBJECT, ENV};

struct Contact
{
    Contact(int body_id1_, int body_id2_, int body_idx1_, int body_idx2_,
            BodyType body_type1_, BodyType body_type2_,
            const double* pos_, const double* frame_)
            : body_id1(body_id1_), body_id2(body_id2_),
              body_idx1(body_idx1_), body_idx2(body_idx2_),
              body_type1(body_type1_), body_type2(body_type2_)
    {
        /**
         * NOTE: 
         * Since Mujoco uses x-axis as the normal. We transformed so that
         * the contact is [y,z,x]
         * Also, Mujoco seems to store the axis in the rows.
         * ref: https://www.roboti.us/forum/index.php?threads/something-wrong-when-return-contact-frame-and-contact-force.3348/
         */
        pos[0] = pos_[0]; pos[1] = pos_[1]; pos[2] = pos_[2];
        frame[0] = frame_[3]; frame[1] = frame_[6]; frame[2] = frame_[0];
        frame[3] = frame_[4]; frame[4] = frame_[7]; frame[5] = frame_[1];
        frame[6] = frame_[5]; frame[7] = frame_[8]; frame[8] = frame_[2];

        pos_mat_to_transform(pos, frame, eigen_frame);
        eigen_pos = eigen_frame.block<3,1>(0,3);
    }

    int body_id1;
    int body_id2;  // contact: body1->body2 (force and vel pointing this way)

    int body_idx1;
    int body_idx2;  // for object that has name object_i, we have i as the idx. Otherwise this is -1

    BodyType body_type1;
    BodyType body_type2;

    double pos[3]; // contact point position in the world frame
    double frame[9]; // contact frame in the world frame. normal is b1->b2 (z-axis)
    /* NOTE: Mujoco has the normal vector as the x-axis. We transformed the contact frame
     * so that the z-axis is the normal vector.
     * our frame relative to the Mujoco one is [y,z,x]
    */
    Vector3d eigen_pos;
    Matrix4d eigen_frame;

    double mu = 1.0;  // friction coeff
};

struct Contacts
{
    Contacts(mjModel* m, mjData* d)
    {
        for (int i=0; i<d->ncon; i++)
        {
            int body1 = m->geom_bodyid[d->contact[i].geom1];
            int body2 = m->geom_bodyid[d->contact[i].geom2];
            // check body type
            int root_body1 = m->body_rootid[body1];
            int root_body2 = m->body_rootid[body2];
            BodyType body_type1, body_type2;
            int body_idx1, body_idx2;
            const char* root_name1 = mj_id2name(m, mjOBJ_BODY, root_body1);
            if (strstr(root_name1, "object") != NULL)
            {
                body_type1 = OBJECT;
                const char* underscore_pos = strchr(mj_id2name(m, mjOBJ_BODY, body1), '_');
                body_idx1 = atoi(underscore_pos+1);

            }
            else if (strcmp("workspace", root_name1) == 0)
            {
                body_type1 = ENV; body_idx1 = -1;
            }
            else if (strcmp("world", root_name1) == 0)
            {
                body_type1 = ENV; body_idx1 = -1;
            }
            else
            {
                body_type1 = ROBOT; body_idx1 = -2;
            }
            const char* root_name2 = mj_id2name(m, mjOBJ_BODY, root_body2);
            if (strstr(root_name2, "object") != NULL)
            {
                body_type2 = OBJECT;
                const char* underscore_pos = strchr(mj_id2name(m, mjOBJ_BODY, body2), '_');
                body_idx2 = atoi(underscore_pos+1);

            }
            else if (strcmp("workspace", root_name2) == 0)
            {
                body_type2 = ENV; body_idx2 = -1;
            }
            else if (strcmp("world", root_name2) == 0)
            {
                body_type2 = ENV; body_idx2 = -1;
            }
            else
            {
                body_type2 = ROBOT; body_idx2 = -2;
            }
            /**
             * NOTE:
             * there is a mismatch of how we represent the contact frame relative to the mujoco
             * we use z-axis as normal vector, rather than x-axis used by Mujoco.
             */
            Contact* contact = new Contact(body1, body2, body_idx1, body_idx2,
                                          body_type1, body_type2,
                                          d->contact[i].pos, d->contact[i].frame);
            contacts.push_back(contact);
        }
    }
    ~Contacts()
    {
        for (int i=0; i<contacts.size(); i++)
        {
            delete contacts[i];
        }
    }

    std::vector<Contact*> contacts;
};

/**
 * @brief 
 * 
 * TODO: unit test
 * @param ang_in 
 * @param n_ss_mode 
 * @param ss_mode 
 */

void ang_to_ss_mode(const double ang_in, const int n_ss_mode,
                    std::vector<int>& ss_mode)
{
    ss_mode.resize(n_ss_mode);
    for (int i=0; i<ss_mode.size(); i++) ss_mode[i] = 0;

    // decide on axis. the region is pi/Nc. index in [0,2Nc)
    // obtain the angle of vel
    double ang = std::fmod(ang_in, 2*M_PI);
    if (ang < 0) ang += 2*M_PI;

    // double ang = ang_in % (2*M_PI);
    // [-pi, pi] -> [0,2pi]
    double ss_ang = M_PI / ss_mode.size();
    // first idx
    int axis0 = ang / (ss_ang);
    int idx0 = axis0 / ss_mode.size();
    int sign0 = 1;
    if (idx0 == 1) sign0 = -1;
    axis0 = axis0 % (ss_mode.size());


    // second idx
    int axis1 = ang / (ss_ang) + 1;
    axis1 = axis1 % (2*ss_mode.size());
    int idx1 = axis1 / ss_mode.size();
    int sign1 = 1;
    if (idx1 == 1) sign1 = -1;
    axis1 = axis1 % (ss_mode.size());


    ss_mode[axis0] = sign0;  ss_mode[axis1] = sign1;
}


void vel_to_contact_mode(const Contact* contact,
                         const Vector6d& twist1, const Vector6d& twist2,
                         const int n_ss_mode,
                         int& cs_mode, std::vector<int>& ss_mode)
{
    // if the contact is obj with workspace, then set cs modes and ss modes
    if (((contact->body_type1 == BodyType::OBJECT) && (contact->body_type2 == BodyType::ENV)) ||
        ((contact->body_type1 == BodyType::ENV) && (contact->body_type2 == BodyType::OBJECT)))
    {
        Vector6d twist = twist1 - twist2; // the relative twist in the world frame
        // since the relative twist is in the world frame, the relative vel at the contact point is:
        // w_omega cross g_wc + w_v
        Vector3d w_vel = twist.head<3>() + twist.tail<3>().cross(contact->eigen_pos);
        // obtain the velocity in the contact frame. g_cw * w_vel
        Vector3d c_vel = contact->eigen_frame.inverse().block<3,3>(0,0) * w_vel;
        ang_to_ss_mode(std::atan2(c_vel[1], c_vel[0]), n_ss_mode, ss_mode);

        cs_mode = 0;
        return;
    }
    // TODO: obj-obj case?

    ss_mode.resize(n_ss_mode);
    for (int i=0; i<ss_mode.size(); i++) ss_mode[i] = 0;
    // robot-obj case: cs_mode = 0, ss_mode=sticking
    if (((contact->body_type1 == BodyType::ROBOT) && (contact->body_type2 == BodyType::OBJECT)) ||
        ((contact->body_type1 == BodyType::OBJECT) && (contact->body_type2 == BodyType::ROBOT)))
    {
        cs_mode = 0;
        return;
    }    

    // otherwise cs_mode = 1
    cs_mode = 1;
}