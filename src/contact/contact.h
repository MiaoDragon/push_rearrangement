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
