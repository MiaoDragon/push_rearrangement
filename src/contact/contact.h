/**
 * construct the "contact" data structure.
 */

#pragma once
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
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
    Contact(int body_id1_, int body_id2_, 
            BodyType body_type1_, BodyType body_type2_,
            const double* pos_, const double* frame_)
            : body_id1(body_id1_), body_id2(body_id2_),
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
            const char* name1 = mj_id2name(m, mjOBJ_BODY, root_body1);
            if (strcmp("object", name1) == 0)
            {
                body_type1 = OBJECT;
            }
            else if (strcmp("workspace", name1) == 0)
            {
                body_type1 = ENV;
            }
            else
            {
                body_type1 = ROBOT;
            }
            const char* name2 = mj_id2name(m, mjOBJ_BODY, root_body2);
            if (strcmp("object", name2) == 0)
            {
                body_type2 = OBJECT;
            }
            else if (strcmp("workspace", name2) == 0)
            {
                body_type2 = ENV;
            }
            else
            {
                body_type2 = ROBOT;
            }
            /**
             * NOTE:
             * there is a mismatch of how we represent the contact frame relative to the mujoco
             * we use z-axis as normal vector, rather than x-axis used by Mujoco.
             */
            Contact* contact = new Contact(body1, body2, 
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
 * Given cs mode and ss mode, generate the matrix to represent the constraint.
 * matrix involved:
 * - equality constraints for v_c
 * - equality/inequality constraint for v_c given cs_mode
 * - equality/inequality constraint 
 *
 * NOTE:
 * this is a general formulation of the constraints.
 * variables: V1, V2 (twist in body frame),
 *            vc (linear velocity at contact, written in contact frame)
 *               body 1 relative to body 2
 *            fc (linear force at contact, written in contact frame)
 * 
 * ss_mode refers to the tangent polyhedron approx to the friction cone
 * @param contact 
 * @param cs_mode 
 * @param ss_mode 
 */
void ContactConstraint(mjModel* m, mjData* d,
                       Contact* contact, int cs_mode, std::vector<int> ss_mode,
                       MatrixXd& Ce, VectorXd& ce0,
                       MatrixXd& Ci, VectorXd& ci0)
{
    Ce.resize(6+2*ss_mode.size(),18);
    Ci.resize(6+2*ss_mode.size(),18);

    /* decision variable: V1, V2, vc, fc */
    // obtain the pose of the two objects
    Matrix4d g1, g2;
    mj_to_transform(m, d, contact->body_id1, g1);
    mj_to_transform(m, d, contact->body_id2, g2);
    // obtain g1 in contact, g2 in contact
    Matrix4d cg1, cg2, cw;
    pos_mat_to_transform(contact->pos, contact->frame, cw);
    cw = cw.inverse();
    cg1 = cw * g1;
    cg2 = cw * g2;

    int ce_padding = 0, ci_padding = 0;  // to record how many constraints we have
    
    /* constraints for Adj_g1[0:3,:] V1 - Adj_g2[0:3,:] V2 - vc=0 */
    // this is Ce[0:3,0:6], Ce[0:3,6:12], Ce[0:3,12:15]
    Matrix6d adj1, adj2;
    adjoint(cg1, adj1); adjoint(cg2, adj2);
    Ce.block<3,6>(0,0) = adj1.block<3,6>(0,0);
    Ce.block<3,6>(0,6) = -adj2.block<3,6>(0,0);
    Ce(0,12+0) = -1; Ce(1,12+1) = -1; Ce(2,12+2) = -1;

    ce_padding += 3;

    /* constraints according to cs mode */
    // - cs mode = 1: not in contact (or breaking contact)
    //   vc dot normal <= 0, fc = 0
    // => vc[2] <= 0, fc=0
    //   velocity: this is Ci[0,12:15]
    //   force: this is Ce[3:6,15:18]

    // - cs mode = 0: in contact
    //   vc dot normal = 0, fc dot normal >= 0
    // => vc[2] = 0, fc[2] >= 0
    // normal = [0,0,1] in the contact frame


    if (cs_mode == 1)  // not in contact
    {
        Ci(ci_padding+0,12+2) = -1;
        Ce(ce_padding+0,15+0) = 1; Ce(ce_padding+1,15+1) = 1; Ce(ce_padding+2,15+2) = 1;
        ce_padding += 3;
        ci_padding += 1;
    }
    else  // in contact
    {
        Ce(ce_padding+0,12+2) = 1;
        Ci(ci_padding+0,15+2) = 1;
        ce_padding += 1; ci_padding += 1;

        /* constraints according to ss mode */
        // number of ss mode partition the space into 2*K regions
        // assuming the i-th one is [cos(180/K*i), sin(180/K*i), 0]
        // - sticking: vt = 0 (vt[1]=vt[2]=0)    (could also use below ones)
        // - ss_mode[i] = +1: vt dot axis[i] >= 0, ft dot axis[i] <= 0
        // - ss_mode[i] = 0: vt dot axis[i] = 0
        // - ss_mode[i] = -1: vt dot axis[i] <= 0, ft dot axis[i] >= 0
        int K = ss_mode.size();
        for (int i=0; i<ss_mode.size(); i++)
        {
            if (ss_mode[i] == 1)
            {
                Ci(ci_padding+0,12+0) = cos(M_PI/K*i);
                Ci(ci_padding+0,12+1) = sin(M_PI/K*i);  // velocity
                Ci(ci_padding+0,15+0) = -cos(M_PI/K*i);
                Ci(ci_padding+0,15+1) = -sin(M_PI/K*i);  // force
                ci_padding += 2;

            }
            else if (ss_mode[i] == -1)
            {
                Ci(ci_padding+0,12+0) = -cos(M_PI/K*i);
                Ci(ci_padding+0,12+1) = -sin(M_PI/K*i);
                Ci(ci_padding+0,15+0) = cos(M_PI/K*i);
                Ci(ci_padding+0,15+1) = sin(M_PI/K*i);
                ci_padding += 2;
            }
            else if (ss_mode[i] == 0)
            {
                Ce(ce_padding+0,12+0) = cos(M_PI/K*i);
                Ce(ce_padding+0,12+1) = sin(M_PI/K*i);
                Ce(ce_padding+0,15+0) = cos(M_PI/K*i);
                Ce(ce_padding+0,15+1) = sin(M_PI/K*i);
                ce_padding += 2;
            }
        }
    
    }



}