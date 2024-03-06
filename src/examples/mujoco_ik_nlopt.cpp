#include <math.h>
#include <nlopt.h>
#include <stdio.h>

#include <mujoco/mjmodel.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjvisualize.h>
#include <vector>
#include <iostream>

extern mjModel* m;                  // MuJoCo model
// extern mjData* d;                   // MuJoCo data
mjData* dsim;


void simulate(std::vector<const char*>& joint_names, const double* q, const char* link_name, double pose[7])
// pose: x y z qw qx qy qz
{
    // get the joint ids of the joint names
    for (int i=0; i<joint_names.size(); i++)
    {
        int joint_idx = mj_name2id(m, mjOBJ_JOINT, joint_names[i]);
        int jnt_qpos = m->jnt_qposadr[joint_idx];
        dsim->qpos[jnt_qpos] = q[i];
    }
    mj_forward(m, dsim);

    // get the link id
    int body_id = mj_name2id(m, mjOBJ_BODY, link_name);

    // get the pose    
    pose[0] = dsim->xpos[body_id*3+0];
    pose[1] = dsim->xpos[body_id*3+1];
    pose[2] = dsim->xpos[body_id*3+2];
    pose[3] = dsim->xquat[body_id*4+0];
    pose[4] = dsim->xquat[body_id*4+1];
    pose[5] = dsim->xquat[body_id*4+2];
    pose[6] = dsim->xquat[body_id*4+3];

}


typedef struct 
{
    std::vector<const char*> joint_names;
    const char* link_name;
    double target_pose[7];
} cost_data;


double costfunc(unsigned n, const double *x, double *grad, void *cost_func_data)
{
    const double alpha = 0.1;
    cost_data *data = (cost_data *) cost_func_data;
    double pose[7] = {0};
    simulate(data->joint_names, x, data->link_name, pose);

    double residual[6];
    mju_subQuat(residual+3, pose+3, data->target_pose+3);

    mju_sub3(residual, pose, data->target_pose);

    double res = 0;
    res = sqrt(residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2]);
    res += alpha * sqrt(residual[3]*residual[3] + residual[4]*residual[4] + residual[5]*residual[5]);
    return res;
}

// typedef struct {
//     double a, b;
// } my_constraint_data;


// typedef struct {
//     double ceq_1;
//     double ceq_2;
// }myequalityconstraints_data;

// typedef struct {
//     double cin_1;
// }myinequalityconstraints_data;


// double myconstraint(unsigned n, const double *x, double *grad, void *data)
// {
//     my_constraint_data *d = (my_constraint_data *) data;
//     double a = d->a, b = d->b;
//     if (grad) {
//         grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
//         grad[1] = -1.0;
//     }
//     return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
//  }


// double myequalityconstraints(unsigned m, double *result, unsigned n,
//                              const double *x,  double *grad,
//                              void *equalitydata)
// {
//     myequalityconstraints_data *data = (myequalityconstraints_data *) equalitydata;

//     double c1 = data->ceq_1;
//     double c2 = data->ceq_2;
//     double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4];
//     result[0] = x1+x2+x3-c1; //5;
//     result[1] = x3*x3+x4-c2; //2;
//  }

//  double myinequalityconstraints(unsigned m, double *result, unsigned n,
//                                 const double *x,  double *grad,
//                                 void* inequalitydata)
//  {
//      myinequalityconstraints_data *data = (myinequalityconstraints_data *) inequalitydata;

//      double c1 = data->cin_1;
//      double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4];
//      result[0] = x4*x4+x5*x5-c1; //5;
//   }


void inverse_kinematics(std::vector<const char*>& joint_names, double* q, const char* link_name, double target_pose[7],
                        double* lb, double* ub)
{

    dsim = mj_makeData(m);
    mj_resetData(m, dsim);
    mj_forward(m, dsim);

    // //establish sizes
    unsigned n = 8; //number of decision variables
    // unsigned m_eq = 2; //number of equality constraints
    // unsigned m_in = 1; //number of inequality constraints


    // double lb[8] = { -1.58, -3.13, -1.9, -2.95, -2.36, -3.13, -1.9, -3.13 }; /* lower bounds */
    // double ub[8] = { 1.58, 3.13, 1.9, 2.95, 2.36, 3.13, 1.9, 3.13 }; /* upper bounds */

    nlopt_opt opt;

    opt = nlopt_create(NLOPT_LN_COBYLA, n); /* algorithm and dimensionality */
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    cost_data data;
    data.joint_names = joint_names;
    data.link_name = link_name;
    for (int i=0; i<7; i++)
    {
        data.target_pose[i] = target_pose[i];
    }
    
    nlopt_set_min_objective(opt, costfunc, &data);

    // my_constraint_data data[2] = { {2,0}, {-1,1} };

    // nlopt_add_inequality_constraint(opt, myconstraint, &data[0], 1e-8);
    // nlopt_add_inequality_constraint(opt, myconstraint, &data[1], 1e-8);

    nlopt_set_xtol_rel(opt, 1e-4);


    // double x[2] = { 1.234, 5.678 };  /* `*`some` `initial` `guess`*` */
    double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
    if (nlopt_optimize(opt, q, &minf) < 0) {
        printf("nlopt failed!\n");
    }
    else {
        printf("found minimum at f(%g,%g,%g,%g,%g,%g,%g,%g) = %0.10g\n", 
                q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], minf);
    }

    nlopt_destroy(opt);

}