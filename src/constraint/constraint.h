/**
 * @file constraint.h
 * @author your name (you@domain.com)
 * @brief 
 * Implement the constraint functions
 * @version 0.1
 * @date 2024-03-07
 * 
 * @copyright Copyright (c) 2024
 * 
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
#include "../contact/contact.h"


/**
 * @brief 
 * Given cs mode and ss mode, generate the matrix to represent the constraint.
 * matrix involved:
 * - equality constraints for v_c
 * - equality/inequality constraint for v_c given cs_mode
 * - equality/inequality constraint 
 * - part of force balance constraint
 *
 * NOTE:
 * this is a general formulation of the constraints.
 * variables: V1, V2 (twist in body or spatial frame),
 *            vc (linear velocity at contact, written in contact frame)
 *               body 1 relative to body 2
 *            fc (linear force at contact, written in contact frame)
 * 
 * ss_mode refers to the tangent polyhedron approx to the friction cone
 * @param contact 
 * @param cs_mode: 1 means NOT in contact. 0 means in contact.
 * @param ss_mode 
 */
void contact_constraint(mjModel* m, mjData* d, Contact* contact, 
                             const int cs_mode, const std::vector<int> ss_mode,
                             MatrixXd& Ce, VectorXd& ce0, int& ce_size,
                             MatrixXd& Ci, VectorXd& ci0, int& ci_size,
                             MatrixXd& Fe1, MatrixXd& Fe2,
                             MatrixXd& Te1, MatrixXd& Te2)
{
    int K = ss_mode.size();
    Ce.resize(2*(6+4*K),6+6+2+K*4);
    Ci.resize(2*(6+4*K),6+6+2+K*4);
    Ce.setZero(); Ci.setZero();
    ce0.resize(2*(6+4*K));
    ci0.resize(2*(6+4*K));

    Fe1.resize(3,6+6+2+K*4); Fe2.resize(3,6+6+2+K*4);
    Te1.resize(3,6+6+2+K*4); Te2.resize(3,6+6+2+K*4);
    Fe1.setZero(); Fe2.setZero();
    Te1.setZero(); Te2.setZero();
    /**
     * decision variable: 
     *  V1, V2, 
     *  vn, vt1^+, vt2^+, ..., vtk^+, vt1^-, vt2^-, ..., vtk^-, (2*k+1)
     *  fn, ft1^+, ft2^+, ..., ftk^+, ft1^-, ft2^-, ..., ftk^-, (2*k+1)
     */

    // obtain the pose of the two objects
    Matrix4d g1, g2;
    mj_to_transform(m, d, contact->body_id1, g1);
    mj_to_transform(m, d, contact->body_id2, g2);

    std::cout << "g1: " << std::endl;
    std::cout << g1 << std::endl;

    std::cout << "g2: " << std::endl;
    std::cout << g2 << std::endl;


    // obtain g1 in contact, g2 in contact
    Matrix4d cg1, cg2, wc, cw;
    pos_mat_to_transform(contact->pos, contact->frame, wc);
    std::cout << "wc: " << std::endl;
    std::cout << wc << std::endl;
    cw = wc.inverse();

    std::cout << "cw: " << std::endl;
    std::cout << cw << std::endl;
    cg1 = cw * g1;
    cg2 = cw * g2;

    int ce_padding = 0, ci_padding = 0;  // to record how many constraints we have

    /* vn *<=* 0, v_tk >= 0, v_tk' >= 0, fn >= 0, f_ti >= 0, f_ti' >= 0*/
    Ci(ci_padding+0, 6*2) = -1;  ci_padding += 1;
    for (int i=0; i<K; i++)
    {
        Ci(ci_padding+0, 6*2+1+i) = 1;  ci_padding += 1;
        Ci(ci_padding+0, 6*2+1+K+i) = 1;  ci_padding += 1;
    }
    Ci(ci_padding+0, 6*2+1+K*2) = 1; ci_padding += 1;
    for (int i=0; i<K; i++)
    {
        Ci(ci_padding+0, 6*2+1+K*2+1+i) = 1; ci_padding += 1;
        Ci(ci_padding+0, 6*2+1+K*2+1+K+i) = 1;  ci_padding += 1;
    }


    /* constraints for Adj_g1[0:3,:] V1 - Adj_g2[0:3,:] V2 - vc=0 */
    // vc = vn * normal + sum_iv_ti1*axis(i) + sum_iv_ti2*(-axis(i))
    // this is Ce[0:3,0:6], Ce[0:3,6:12], Ce[0:3,12:15]
    /**
     * @brief 
     * depends on what coord frame V1 and V2 are in. The difference affects adjoint.
     * If we specify V1, V2 in their body frame, the adjoint should be Ad_gc1 and Ad_gc2.
     * If we specify V1, V2 in their spatial frame, then adjoint should be Ad_gcw.
     * NOTE:
     * for easier case, since Mujoco Jacobian should be in spatial frame, we use spatial frame.
     */

    Matrix6d adj1, adj2, adj;
    adjoint(cg1, adj1); adjoint(cg2, adj2);
    adjoint(cw, adj);
    std::cout << "adj1: " << std::endl;
    std::cout << adj1 << std::endl;
    std::cout << "adj2: " << std::endl;
    std::cout << adj2 << std::endl;
    std::cout << "adj: " << std::endl;
    std::cout << adj << std::endl;

    // Ce.block<3,6>(ce_padding+0,0) = adj1.block<3,6>(0,0);
    // Ce.block<3,6>(ce_padding+0,6) = -adj2.block<3,6>(0,0);
    Ce.block<3,6>(ce_padding+0,0) = adj.block<3,6>(0,0);
    Ce.block<3,6>(ce_padding+0,6) = -adj.block<3,6>(0,0);

    Ce(ce_padding+2,6*2) = -1;  // -normal
    for (int i=0; i<K; i++)
    {
         // -v_ti1*axis(i)
        Ce(ce_padding+0,6*2+1+i) = -cos(M_PI/K*i);
        Ce(ce_padding+1,6*2+1+i) = -sin(M_PI/K*i);
        // -v_ti2*(-axis(i))
        Ce(ce_padding+0,6*2+1+K+i) = cos(M_PI/K*i);
        Ce(ce_padding+1,6*2+1+K+i) = sin(M_PI/K*i);
    }
    ce_padding += 3;


    /* constraints according to cs mode */
    // - cs mode = 1: not in contact (or breaking contact)
    //   vc dot normal <= 0, fc = 0
    //   (vn <= 0, fn = 0)
    //   velocity: this is Ci[0,12:15]
    //   force: this is Ce[3:6,15:18]

    // - cs mode = 0: in contact
    //   vc dot normal = 0, fc dot normal >= 0
    // => vn = 0, fn >= 0
    // normal = [0,0,1] in the contact frame


    if (cs_mode == 1)  // not in contact
    {
        Ci(ci_padding+0,6*2) = -1;  ci_padding+=1;
        Ce(ce_padding+0,6*2+1+K*2) = 1;  ce_padding+=1;
    }
    else  // in contact
    {
        Ce(ce_padding+0,6*2) = 1;  ce_padding+=1;
        Ci(ci_padding+0,6*2+1+K*2) = 1;  ci_padding+=1;

        /* constraints according to ss mode */
        // number of ss mode partition the space into 2*K regions
        // assuming the i-th one is [cos(180/K*i), sin(180/K*i), 0]
        // - sticking: vt = 0 (v_t1=0,...)    (could also use below ones)
        //   sum_i ft_i <= mu*fn (inside polyhedral cone)
 
        // - ss_mode[i] = +1: vt dot axis[i] >= 0, ft dot axis[i] >= 0  (all is body1 -> body2)
        // - ss_mode[i] = 0: vt dot axis[i] = 0, ft dot axis[i] = 0
        // - ss_mode[i] = -1: vt dot axis[i] <= 0, ft dot axis[i] <= 0
        // - sliding: sum(ft_i^+) + sum(ft_i^-) = mu*fn. 

        /**
         * @brief 
         * NOTE:
         * body 1 relative to body 2 is in tangent vel dir 1, then body1 experiences
         * friction force from body 2 in the opposite dir of 1. But body 2 experiences
         * friction force in the same direction as dir 1.
         */
        bool is_sticking = true;
        for (int i=0; i<ss_mode.size(); i++)
        {
            if (ss_mode[i] != 0)
            {
                is_sticking = false;
                break;
            }
        }

        if (is_sticking)
        {
            // vt = 0.: Ce*v=0
            // Ce(ce_padding+0,12+0) = 1;
            // Ce(ce_padding+1,12+1) = 1;
            // ce_padding += 2;

            for (int i=0; i<ss_mode.size(); i++)
            {
                // v_ti=0, v_ti'=0
                Ce(ce_padding+0, 6*2+1+i) = 1;  ce_padding += 1;
                Ce(ce_padding+0, 6*2+1+K+i) = 1;  ce_padding += 1;
            }

            // sum_i f_ti + sum_i f_ti' <= mu*f_n
            // => sum_i f_ti + sum_i f_ti' - mu*f_n <= 0
            // => sum_i -f_ti + sum_i -f_ti' + mu*f_n >= 0

            Ci(ci_padding+0, 2*6+1+2*K) = contact->mu;
            for (int i=0; i<ss_mode.size(); i++)
            {
                Ci(ci_padding+0, 2*6+1+2*K+1+i) = -1;
                Ci(ci_padding+0, 2*6+1+2*K+1+K+i) = -1;
            }
            ci_padding += 1;
        }
        else
        {
            // sliding: sum_i f_ti + sum_i f_ti' = fn*mu
            // => sum_i f_ti + sum_i f_ti' - fn*mu = 0
            Ce(ce_padding+0,6*2+1+2*K) = -contact->mu;
            for (int i=0; i<ss_mode.size(); i++)
            {
                Ce(ce_padding+0,6*2+1+2*K+1+i) = 1;
                Ce(ce_padding+0,6*2+1+2*K+1+K+i) = 1;
            }
            ce_padding += 1;

            for (int i=0; i<ss_mode.size(); i++)
            {
                if (ss_mode[i] == 1)
                {
                    // v_ti >= 0, v_ti' = 0
                    // f_ti >= 0, f_ti' = 0
                    Ci(ci_padding+0,6*2+1+i) = 1; ci_padding += 1;
                    Ci(ci_padding+0,6*2+1+2*K+1+i) = 1; ci_padding += 1;
                    Ce(ce_padding+0,6*2+1+K+i) = 1; ce_padding += 1;
                    Ce(ce_padding+0,6*2+1+2*K+1+K+i) = 1; ce_padding += 1;
                }
                else if (ss_mode[i] == -1)
                {
                    // v_ti' >= 0, f_ti' >= 0
                    // v_ti = 0, f_ti = 0
                    Ci(ci_padding+0,6*2+1+K+i) = 1; ci_padding += 1;
                    Ci(ci_padding+0,6*2+1+2*K+1+K+i) = 1; ci_padding += 1;
                    Ce(ce_padding+0,6*2+1+i) = 1; ce_padding += 1;
                    Ce(ce_padding+0,6*2+1+2*K+1+i) = 1; ce_padding = 1;
                }
                else if (ss_mode[i] == 0)
                {
                    // v_ti = 0, f_ti = 0, v_ti'=0, f_ti'=0
                    Ce(ce_padding+0,6*2+1+i) = 1; ce_padding += 1;
                    Ce(ce_padding+0,6*2+1+K+i) = 1; ce_padding += 1;
                    Ce(ce_padding+0,6*2+1+2*K+1+i) = 1; ce_padding += 1;
                    Ce(ce_padding+0,6*2+1+2*K+1+K+i) = 1; ce_padding += 1;
                }
            }
        }
    }

    ce_size = ce_padding;
    ci_size = ci_padding;

    // std::cout << "contact->b1_idx: " << contact->body_idx1 << std::endl;
    // std::cout << "contact->b2_idx: " << contact->body_idx2 << std::endl;

    // std::cout << "ce_size: " << ce_size << std::endl;
    // std::cout << "ci_size: " << ci_size << std::endl;

    std::cout << "after adding contact matrix" << std::endl;
    Ce.conservativeResize(ce_size,6+6+2+K*4);  Ci.conservativeResize(ci_size,6+6+2+K*4);
    std::cout << "Ce: " << std::endl;
    std::cout << Ce << std::endl;
    std::cout << "Ci: " << std::endl;
    std::cout << Ci << std::endl;


    /* force balance constraint */
    // w_R_c c_f_c^{b} (add this to the force balance constraint)
    // => w_R_c (f_n*[0,0,1]+sum_i f_ti*axis_i-sum_i f_ti'*axis_i)

    // for body 1, force is (-c_f_c^{b})
    Vector3d axis(0,0,1);
    Matrix3d wRc = wc.block<3,3>(0,0);

    Fe1.block<3,1>(0,6*2+1+2*K) = -wRc * axis;

    for (int i=0; i<K; i++)
    {
        axis[0] = cos(M_PI/K*i); axis[1] = sin(M_PI/K*i); axis[2] = 0;
        Fe1.block<3,1>(0,6*2+1+2*K+1+i) = -wRc * axis;
        axis[0] = -cos(M_PI/K*i); axis[1] = -sin(M_PI/K*i); axis[2] = 0;
        Fe1.block<3,1>(0,6*2+1+2*K+1+K+i) = -wRc * axis;
    }

    // for body 2, force is c_f_c^{b}
    axis[0] = 0; axis[1] = 0; axis[2] = 1;
    Fe2.block<3,1>(0,6*2+1+2*K) = wRc * axis;
    for (int i=0; i<K; i++)
    {
        axis[0] = cos(M_PI/K*i); axis[1] = sin(M_PI/K*i); axis[2] = 0;
        Fe2.block<3,1>(0,6*2+1+2*K+1+i) = wRc * axis;
        axis[0] = -cos(M_PI/K*i); axis[1] = -sin(M_PI/K*i); axis[2] = 0;
        Fe2.block<3,1>(0,6*2+1+2*K+1+K+i) = wRc * axis;
    }


    /* torque balance constraint */
    // (pos_c - COM_b) cross (w_R_c c_f_c^{b})
    // => (pos_c - COM_b) cross (w_R_c (f_n*[0,0,1]+sum_i f_ti*axis_i-sum_i f_ti'*axis_i))

    // for body 1, force is -(c_f_c^b)
    Vector3d r;
    r[0] = wc(0,3) - d->xipos[contact->body_id1*3+0];
    r[1] = wc(1,3) - d->xipos[contact->body_id1*3+1];
    r[2] = wc(2,3) - d->xipos[contact->body_id1*3+2];
    std::cout << "body 1, r: " << std::endl;
    std::cout << r << std::endl;

    Matrix3d r_hat;
    hat_operator(r, r_hat);
    axis[0] = 0; axis[1] = 0; axis[2] = 1;
    Te1.block<3,1>(0,2*6+1+K*2) = -r_hat * wRc * axis;
    for (int i=0; i<K; i++)
    {
        axis[0] = cos(M_PI/K*i); axis[1] = sin(M_PI/K*i); axis[2] = 0;
        Te1.block<3,1>(0,2*6+1+K*2+1+i) = -r_hat * wRc * axis;
        axis[0] = -cos(M_PI/K*i); axis[1] = -sin(M_PI/K*i); axis[2] = 0;
        Te1.block<3,1>(0,2*6+1+K*2+1+K+i) = -r_hat * wRc * axis;
    }

    // for body2, force is (c_f_c^b)
    r[0] = wc(0,3) - d->xipos[contact->body_id2*3+0];
    r[1] = wc(1,3) - d->xipos[contact->body_id2*3+1];
    r[2] = wc(2,3) - d->xipos[contact->body_id2*3+2];
    hat_operator(r, r_hat);
    std::cout << "body 2, r: " << std::endl;
    std::cout << r << std::endl;

    axis[0] = 0; axis[1] = 0; axis[2] = 1;
    Te2.block<3,1>(0,2*6+1+K*2) = r_hat * wRc * axis;
    for (int i=0; i<K; i++)
    {
        axis[0] = cos(M_PI/K*i); axis[1] = sin(M_PI/K*i); axis[2] = 0;
        Te2.block<3,1>(0,2*6+1+K*2+1+i) = r_hat * wRc * axis;
        axis[0] = -cos(M_PI/K*i); axis[1] = -sin(M_PI/K*i); axis[2] = 0;
        Te2.block<3,1>(0,2*6+1+K*2+1+K+i) = r_hat * wRc * axis;
    }

    std::cout << "wc: " << std::endl;
    std::cout << wc << std::endl;

    std::cout << "Fe1: " << std::endl;
    std::cout << Fe1 << std::endl;

    std::cout << "Fe2: " << std::endl;
    std::cout << Fe2 << std::endl;

    std::cout << "Te1: " << std::endl;
    std::cout << Te1 << std::endl;

    std::cout << "Te2: " << std::endl;
    std::cout << Te2 << std::endl;


}


/**
 * @brief 
 * copy over the constraints at contact, to create matrix for optimization.
 * 
 * @param m 
 * @param d 
 * @param robot_v_indices 
 * @param obj_body_indices 
 * @param contacts 
 * @param cs_modes 
 * @param ss_modes 
 * @param Ae 
 * @param ae0 
 * @param ae_size 
 * @param Ai 
 * @param ai0 
 * @param ai_size 
 */
void total_constraints(mjModel* m, mjData* d, std::vector<int>& robot_v_indices,
                       std::vector<int>& obj_body_indices,
                       Contacts& contacts, 
                       const std::vector<int> cs_modes, 
                       const std::vector<std::vector<int>> ss_modes,
                       MatrixXd& Ae, VectorXd& ae0, int& ae_size,
                       MatrixXd& Ai, VectorXd& ai0, int& ai_size
                       )
{
    /* decision variables: q, v0, v1, v2, ..., vn, v^i_c_n, v^i_c_t1, ..., v^i_c_tk, f^i_c_n, f^i_c_t1, ..., f^i_c_tk, ...*/
    /* summary: q (nq), vi (6*nobj), contact (vel: 1+2K, force: 1+2K)*/

    int nq = robot_v_indices.size();
    int nobj = obj_body_indices.size();
    int ncon = cs_modes.size();
    int K = ss_modes[0].size();
    MatrixXd Fe(nobj*3,nq+nobj*6+ncon*2*(1+2*K)), 
             Te(nobj*3,nq+nobj*6+ncon*2*(1+2*K)); // force balance and torque balance matrix
                                              // for each object, the force/torque has dim 3
    VectorXd fe0(nobj*3), te0(nobj*3);
    Fe.setZero(); Te.setZero(); fe0.setZero(); te0.setZero();

    Ae.resize((6+4*K)*ncon+2*nobj*3,nq+nobj*6+ncon*2*(1+2*K));
    Ai.resize((6+4*K)*ncon+2*nobj*3,nq+nobj*6+ncon*2*(1+2*K));
    ae0.resize((6+4*K)*ncon+2*nobj*3);
    ai0.resize((6+4*K)*ncon+2*nobj*3);
    Ae.setZero(); Ai.setZero(); ae0.setZero(); ai0.setZero();

    int ae_offset=0, ai_offset=0;

    std::cout << "nobj: " << nobj << std::endl;

    for (int i=0; i<cs_modes.size(); i++)
    {
        MatrixXd Ce, Ci;
        VectorXd ce0, ci0;
        int ce_size, ci_size;
        MatrixXd Fe1, Fe2, Te1, Te2;
        contact_constraint(m, d, contacts.contacts[i], cs_modes[i], ss_modes[i], 
                           Ce, ce0, ce_size, Ci, ci0, ci_size, 
                           Fe1, Fe2, Te1, Te2);

        // if the body is an object, then put values in the fields of Ae that corresponds to body1
        // if the body is an env, then it doesn't exist in the devision var, so the related fields not be copied.
        // if the body is a robot, then we should use inverse kinematics
        int body_idx1 = contacts.contacts[i]->body_idx1;
        int body_idx2 = contacts.contacts[i]->body_idx2;
        if (body_idx1 >= 0)
        {
            // object: directly copy the matrix over
            int idx = contacts.contacts[i]->body_idx1;
            Ae.block(ae_offset, nq+idx*6, ce_size, 6) = Ce.block(0, 0, ce_size, 6);
            Ai.block(ai_offset, nq+idx*6, ci_size, 6) = Ci.block(0, 0, ci_size, 6);
        }
        else if (body_idx1 == -2)
        {
            // robot: need to multiply by the inverse of jacobian
            // TODO
            // assuming Mujoco jacobian is in the world coord system. 
            // original constraint is: AV = 0.
            // change to robot config, it is: AJq=0
            double jacp[3*m->nv], jacr[3*m->nv];

            mj_jacBody(m, d, jacp, jacr, contacts.contacts[i]->body_id1);
            MatrixXd J_s(6,nq);  // in spatial frame assumed. Shape: 6xnv
            for (int j=0; j<nq; j++)
            {
                J_s(0,j) = jacp[0*m->nv+robot_v_indices[j]];
                J_s(1,j) = jacp[1*m->nv+robot_v_indices[j]];
                J_s(2,j) = jacp[2*m->nv+robot_v_indices[j]];
                J_s(3,j) = jacr[0*m->nv+robot_v_indices[j]];
                J_s(4,j) = jacr[1*m->nv+robot_v_indices[j]];
                J_s(5,j) = jacr[2*m->nv+robot_v_indices[j]];
            }
            std::cout << "Jacobian: " << std::endl;
            std::cout << J_s << std::endl;
            Ae.block(ae_offset, 0, ce_size, nq) = Ce.block(0, 0, ce_size, 6) * J_s;
            Ai.block(ai_offset, 0, ci_size, nq) = Ci.block(0, 0, ci_size, 6) * J_s;
        }

        if (body_idx2 >= 0)
        {
            // object: directly copy the matrix over
            int idx = contacts.contacts[i]->body_idx2;
            Ae.block(ae_offset, nq+idx*6, ce_size, 6) = Ce.block(0, 6, ce_size, 6);
            Ai.block(ai_offset, nq+idx*6, ci_size, 6) = Ci.block(0, 6, ci_size, 6);
        }
        else if (body_idx2 == -2)
        {
            // robot: need to multiply by the inverse of jacobian
            // TODO
            double jacp[3*m->nv], jacr[3*m->nv];

            mj_jacBody(m, d, jacp, jacr, contacts.contacts[i]->body_id2);
            MatrixXd J_s(6,nq);  // in spatial frame assumed. Shape: 6xnv
            for (int j=0; j<nq; j++)
            {
                J_s(0,j) = jacp[0*m->nv+robot_v_indices[j]];
                J_s(1,j) = jacp[1*m->nv+robot_v_indices[j]];
                J_s(2,j) = jacp[2*m->nv+robot_v_indices[j]];
                J_s(3,j) = jacr[0*m->nv+robot_v_indices[j]];
                J_s(4,j) = jacr[1*m->nv+robot_v_indices[j]];
                J_s(5,j) = jacr[2*m->nv+robot_v_indices[j]];
            }
            std::cout << "Jacobian: " << std::endl;
            std::cout << J_s << std::endl;
            Ae.block(ae_offset, 0, ce_size, nq) = Ce.block(0, 6, ce_size, 6) * J_s;
            Ai.block(ai_offset, 0, ci_size, nq) = Ci.block(0, 6, ci_size, 6) * J_s;
        }


        // handle contact. contact offset: robot_nv + 6*n_obj + i*2*(1+K*2)  [vel: 1+K*2, force: 1+K*2]
        Ae.block(ae_offset, nq+6*nobj+i*2*(1+K*2), ce_size, 2*(1+K*2)) = \
            Ce.block(0, 6*2, ce_size, 2*(1+K*2));
        Ai.block(ai_offset, nq+6*nobj+i*2*(1+K*2), ci_size, 2*(1+K*2)) = \
            Ci.block(0, 6*2, ci_size, 2*(1+K*2));

        ae_offset += ce_size;
        ai_offset += ci_size;


        // handle force balance and torque balance
        if (body_idx1 >= 0)
        {
            // body 1 is object. we add Fe1 into Fe. (copy the contact force cols)
            Fe.block(body_idx1*3,nq+nobj*6+i*2*(1+2*K),3,2*(1+2*K)) = \
                Fe1.block(0,2*6,3,2*(1+2*K));
            Te.block(body_idx1*3,nq+nobj*6+i*2*(1+2*K),3,2*(1+2*K)) = \
                Te1.block(0,2*6,3,2*(1+2*K));
        }
        if (body_idx2 >= 0)
        {
            // body 2 is object. we add Fe2 into Fe.
            Fe.block(body_idx2*3,nq+nobj*6+i*2*(1+2*K),3,2*(1+2*K)) = \
                Fe2.block(0,2*6,3,2*(1+2*K));
            Te.block(body_idx2*3,nq+nobj*6+i*2*(1+2*K),3,2*(1+2*K)) = \
                Te2.block(0,2*6,3,2*(1+2*K));
        }
    }
    // add gravity force to Fe
    for (int i=0; i<nobj; i++)
    {
        fe0[i*3+2] = -m->body_mass[obj_body_indices[i]] * 9.81;
    }

    // integrate force balance and torque balance to Ae and Ai
    Ae.block(ae_offset,0,nobj*3,nq+nobj*6+ncon*2*(1+2*K)) \
        = Fe;
    ae0.segment(ae_offset,nobj*3) = fe0;
    ae_offset += nobj*3;
    Ae.block(ae_offset,0,nobj*3,nq+nobj*6+ncon*2*(1+2*K)) \
        = Te;
    ae_offset += nobj*3;
    ae_size = ae_offset; ai_size = ai_offset;
    // te0 = 0

    Ae.conservativeResize(ae_size,nq+nobj*6+ncon*2*(1+2*K));
    Ai.conservativeResize(ai_size,nq+nobj*6+ncon*2*(1+2*K));
    ae0.conservativeResize(ae_size);
    ai0.conservativeResize(ai_size);
}

/**
 * @brief 
 * objective: 0.5 x^TGx + g0^Tx
 * equivalent: x^TGx + 2*g0^Tx
 * vel objective:
 * sum_i (v(i)-v_target(i))^2 + epsilon_lambda*(w(i)-w_target(i))^2 + epsilon_lambda(q^Tq + sum_i contact_i^Tcontact_i)
 * @param m 
 * @param d 
 * @param robot_v_indices 
 * @param obj_body_indices 
 * @param G 
 * @param g0 
 */
void vel_objective(mjModel* m, mjData* d, 
                   std::vector<int>& robot_v_indices,
                   std::vector<Vector6d>& target_vs,
                   std::vector<int>& active_vs,  // if the vel is active
                   const int ncon, const int K,  // K: ss mode
                   MatrixXd& G, VectorXd& g0)
{
    double epsilon_w = 1, epsilon_lambda = 1e-4;
    int nq = robot_v_indices.size(), nobj = target_vs.size();
    G.resize(nq+nobj*6+ncon*2*(1+2*K),nq+nobj*6+ncon*2*(1+2*K));
    G.setZero();
    g0.resize(nq+nobj*6+ncon*2*(1+2*K));
    g0.setZero();
    VectorXd G_diagonal(nq+nobj*6+ncon*2*(1+2*K));
    G_diagonal.setZero();
    // for each body, set the linear vel part of G to be one
    for (int i=0; i<nobj; i++)
    {
        // if not active, then ignore
        if (active_vs[i] == 0) continue;
        G_diagonal.segment(nq+i*6,3).setOnes();
        G_diagonal.segment(nq+i*6+3,3).setConstant(epsilon_w);
        g0.segment(nq+i*6,3) = -target_vs[i].head(3);
        g0.segment(nq+i*6+3,3) = -epsilon_w*target_vs[i].tail(3);
    }
    // add obj for q, contacts
    // NOTE: need to make sure G is PD
    G_diagonal.segment(0,nq).setConstant(epsilon_lambda);
    G_diagonal.tail(ncon*2*(1+2*K)).setConstant(epsilon_lambda);
    G.diagonal() = G_diagonal;

    std::cout << "G: " << std::endl;
    std::cout << G << std::endl;
    std::cout << "g0: " << std::endl;
    std::cout << g0 << std::endl;
}