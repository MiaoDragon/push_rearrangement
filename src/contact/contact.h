/**
 * construct the "contact" data structure.
 */

#pragma once
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>

#include "../utilities/utilities.h"


enum BodyType {ROBOT, OBJECT, ENV, EE_POSE, EE_POSITION};

struct Contact
{
    Contact(int body_id1_, int body_id2_, int body_idx1_, int body_idx2_,
            BodyType body_type1_, BodyType body_type2_,
            const double* pos_, const double* frame_, const double mu_=1.0)
            : body_id1(body_id1_), body_id2(body_id2_),
              body_idx1(body_idx1_), body_idx2(body_idx2_),
              body_type1(body_type1_), body_type2(body_type2_),
              mu(mu_)
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
    int body_idx2;  // for object that has name object_i, we have i as the idx. Otherwise this is -1 for env, -2 for robot
                    // UPDATE: adding -3 for robot end-effector alone

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


double EE_FRICTION = .1;
double TABLE_FRICTION = 1;

struct Contacts
{
    Contacts(){contacts.resize(0);}
    Contacts(const mjModel* m, const mjData* d)
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
            double friction = 1.0;
            const char* root_name1 = mj_id2name(m, mjOBJ_BODY, root_body1);
            if (strstr(root_name1, "object") != NULL)  // object name: object_[idx]
            {
                body_type1 = OBJECT;
                const char* underscore_pos = strchr(mj_id2name(m, mjOBJ_BODY, body1), '_');
                body_idx1 = atoi(underscore_pos+1);
            }
            else if (strcmp("workspace", root_name1) == 0)
            {
                body_type1 = ENV; body_idx1 = -1; friction = TABLE_FRICTION;
            }
            else if (strcmp("world", root_name1) == 0)
            {
                body_type1 = ENV; body_idx1 = -1;
            }
            else
            {
                body_type1 = ROBOT; body_idx1 = -2; friction = EE_FRICTION;
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
                body_type2 = ENV; body_idx2 = -1; friction = TABLE_FRICTION;
            }
            else if (strcmp("world", root_name2) == 0)
            {
                body_type2 = ENV; body_idx2 = -1;
            }
            else
            {
                body_type2 = ROBOT; body_idx2 = -2; friction = EE_FRICTION;
            }
            /**
             * NOTE:
             * there is a mismatch of how we represent the contact frame relative to the mujoco
             * we use z-axis as normal vector, rather than x-axis used by Mujoco.
             */
            Contact* contact = new Contact(body1, body2, body_idx1, body_idx2,
                                          body_type1, body_type2,
                                          d->contact[i].pos, d->contact[i].frame,
                                          friction);
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
 * @brief define a FocusedContacts structure which focus on a subset
 * of the objects, and use robot end-effector ball (pose or pos)
 * NOTE:
 * (requirement on XML file)
 * Robot link ee_pose refers to when using end-effector pose.
 * Robot link ee_position refers to when using end-effector position.
 * 
 */

struct FocusedContacts : public Contacts
{
  public:
    FocusedContacts(const mjModel* m, const mjData* d, 
                    const std::unordered_set<int>& obj_body_indices,  // (root ids) obj bodies to focus on. The rest is treated as ENV.
                                                               // (TODO: maybe we can use more informed type than ENV)
                    const int& robot_type  // 0: consider robot, 1: consider ee_pose, 2: consider ee_position, -1: don't consider robot
                    ) : Contacts()
    {
        this->robot_type = robot_type;
        this->obj_body_indices = obj_body_indices;
        for (int i=0; i<d->ncon; i++)
        {
            int body1 = m->geom_bodyid[d->contact[i].geom1];
            int body2 = m->geom_bodyid[d->contact[i].geom2];
            // check body type
            int root_body1 = m->body_rootid[body1];
            int root_body2 = m->body_rootid[body2];
            BodyType body_type1=ENV, body_type2=ENV;
            int body_idx1=-1, body_idx2=-1;
            double friction = 1.0;
            const char* root_name1 = mj_id2name(m, mjOBJ_BODY, root_body1);
            std::cout << "contact body 1 geom name:" << mj_id2name(m, mjOBJ_GEOM, d->contact[i].geom1) << std::endl;
            std::cout << "contact body 2 geom name:" << mj_id2name(m, mjOBJ_GEOM, d->contact[i].geom2) << std::endl;
            std::cout << "contact body 1 rootname:" << mj_id2name(m, mjOBJ_BODY, root_body1) << std::endl;
            std::cout << "contact body 2 rootname:" << mj_id2name(m, mjOBJ_BODY, root_body2) << std::endl;

            if (strstr(root_name1, "object") != NULL)  // object name: object_[idx]
            {
                if (obj_body_indices.find(root_body1) != obj_body_indices.end())
                {
                    // found the object inside the focused list
                    body_type1 = OBJECT;
                    const char* underscore_pos = strchr(mj_id2name(m, mjOBJ_BODY, body1), '_');
                    body_idx1 = atoi(underscore_pos+1);
                }
                else
                {
                    // otherwise, treat as env
                    body_type1 = ENV; body_idx1 = -1;
                }
            }
            else if (strcmp("workspace", root_name1) == 0)
            {
                body_type1 = ENV; body_idx1 = -1; friction = TABLE_FRICTION;
            }
            else if (strcmp("world", root_name1) == 0)
            {
                body_type1 = ENV; body_idx1 = -1;
            }
            else if (strcmp("ee_pose", root_name1) == 0)
            {
                if (robot_type != 1)  // we are not using this. Ignore contact
                {
                    continue;
                }
                // consider this as robot
                body_type1 = EE_POSE; body_idx1 = -3; friction = EE_FRICTION;
            }
            else if (strcmp("ee_position", root_name1) == 0)
            {
                if (robot_type != 2)
                {
                    continue;
                }
                body_type1 = EE_POSITION; body_idx1 = -4; friction = EE_FRICTION;
            }
            else // robot
            {
                if (robot_type != 0) // if not using robot, ignore
                {
                    continue;
                }
                body_type1 = ROBOT; body_idx1 = -2; friction = EE_FRICTION;
            }
            const char* root_name2 = mj_id2name(m, mjOBJ_BODY, root_body2);
            std::cout << "contact body 2:" << root_name2 << std::endl;
            if (strstr(root_name2, "object") != NULL)
            {
                if (obj_body_indices.find(root_body2) != obj_body_indices.end())
                {
                    // found the object inside the focused list
                    body_type2 = OBJECT;
                    const char* underscore_pos = strchr(mj_id2name(m, mjOBJ_BODY, body2), '_');
                    body_idx2 = atoi(underscore_pos+1);
                }
                else
                {
                    // otherwise, treat as env
                    body_type1 = ENV; body_idx1 = -1;
                }
            }
            else if (strcmp("workspace", root_name2) == 0)
            {
                body_type2 = ENV; body_idx2 = -1; friction = TABLE_FRICTION;
            }
            else if (strcmp("world", root_name2) == 0)
            {
                body_type2 = ENV; body_idx2 = -1;
            }
            else if (strcmp("ee_pose", root_name2) == 0)
            {
                if (robot_type != 1)  // we are not using this. Ignore contact
                {
                    continue;
                }
                // consider this as robot
                body_type2 = EE_POSE; body_idx2 = -3; friction = EE_FRICTION;
            }
            else if (strcmp("ee_position", root_name2) == 0)
            {
                if (robot_type != 2)
                {
                    continue;
                }
                body_type2 = EE_POSITION; body_idx2 = -4; friction = EE_FRICTION;
            }
            else // robot
            {
                if (robot_type != 0) // if not using robot, ignore
                {
                    continue;
                }
                body_type2 = ROBOT; body_idx2 = -2; friction = EE_FRICTION;
            }

            /**
             * NOTE:
             * there is a mismatch of how we represent the contact frame relative to the mujoco
             * we use z-axis as normal vector, rather than x-axis used by Mujoco.
             */
            Contact* contact = new Contact(body1, body2, body_idx1, body_idx2,
                                          body_type1, body_type2,
                                          d->contact[i].pos, d->contact[i].frame, 
                                          friction);
            contacts.push_back(contact);
        }
    }
    void set_robot_type(const int robot_type) {this->robot_type = robot_type;}
  protected:
    std::unordered_set<int> obj_body_indices;
    int robot_type = -1;
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
                    std::vector<int>& ss_mode);

void vel_to_contact_mode(const Contact* contact,
                         const Vector6d& twist1, const Vector6d& twist2,
                         const int n_ss_mode,
                         int& cs_mode, std::vector<int>& ss_mode);

void vel_to_contact_modes(const Contacts& contacts,
                          const std::unordered_map<int,Vector6d>& twists, // body_id -> twist
                          const int n_ss_mode,
                          std::vector<int>& cs_modes,
                          std::vector<std::vector<int>>& ss_modes);