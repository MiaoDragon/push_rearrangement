#include <algorithm>
#include "mujoco_mppi_intvel.h"
#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include <omp.h>

MujocoMPPIControllerIntvel::MujocoMPPIControllerIntvel(const int& H_in, const int& N_in, const double& default_sigma_in,
                                            const MatrixXd& nominal_x_in,
                                            const MatrixXd& nominal_u_in,
                                            const VectorXd& x_ll_in, const VectorXd& x_ul_in,
                                            const VectorXd& u_ll_in, const VectorXd& u_ul_in)
                                            : MujocoMPPIController(H_in, N_in, default_sigma_in, nominal_u_in,
                                                                   u_ll_in, u_ul_in),
                                              nominal_x(nominal_x_in), x_ll(x_ll_in), x_ul(x_ul_in)
{
    nominal_start_idx = 0;  // reset nominal index
}

void MujocoMPPIControllerIntvel::set_pos_act_indices(const std::vector<int>& pos_act_indices_in)
{
    pos_act_indices = pos_act_indices_in;
}
void MujocoMPPIControllerIntvel::set_vel_ctrl_indices(const std::vector<int>& vel_ctrl_indices_in)
{
    vel_ctrl_indices = vel_ctrl_indices_in;
}


/* first sample controls in truncated gaussian. Then fit a spline to it. Then integrate to obtain state. */
void MujocoMPPIControllerIntvel::sample(std::vector<MatrixXd>& samples)
{
    // sample truncated gaussian to obtain N samples
    samples.resize(N);
    int seed = 12345678;
    for (int i = 0; i<N; i++)
    {
        truncated_gauss(mu, sigma, u_ll, u_ul, seed, samples[i]);
    }
    // TODO: fit spline
}

/**
 * @brief 
 * Given start state as the observed state, sample eps_xs and eps_us
 * TODO: change all x_samples to name eps_x_samples, since we are doing disturbance optimization
 * 
 */
void MujocoMPPIControllerIntvel::sample(const VectorXd& start_state, std::vector<MatrixXd>& x_samples, std::vector<MatrixXd>& u_samples)
{
    // sample truncated gaussian to obtain N samples
    // x_samples.reserve(N);
    // u_samples.reserve(N);

    int seed = 12345678;

    for (int i = 0; i<N; i++)
    {
        MatrixXd u_sample;
        truncated_gauss(mu, sigma, u_ll, u_ul, seed, u_sample);
        // ERROR: this u samples are control limits. We want disturbance limits

        // for (int j=0; j<u_sample.rows(); j++)  
        // {
        //     u_sample(j,0) = 0;
        //     u_sample(j,1) = 0.3;
        //     u_sample(j,2) = 0;
        // }

        u_samples.push_back(u_sample);
    }

    // TODO: fit spline

    // integrate to obtain x samples.
    for (int i=0; i<N; i++)
    {
        int nominal_idx = std::min(nominal_start_idx, static_cast<int>(nominal_x.rows())-1);
        nominal_idx = std::max(nominal_idx, 0);

        MatrixXd x_sample;
        x_sample.resize(H+1,x_ll.size());
        x_sample.row(0) = start_state - nominal_x.row(nominal_idx).transpose();
        for (int j=1; j<H+1; j++)
        {
            x_sample.row(j) = x_sample.row(j-1) + u_samples[i].row(j-1)*dt;
        }
        x_samples.push_back(x_sample);
    }
}


void MujocoMPPIControllerIntvel::step(const mjModel *m, const double* sensordata, VectorXd& control)
{
    // observe: obtain the state vars. for simplicity, assume it's fed in

    double bound_cost_scale = 1.0;


    // sample(N) -> eps  // N x H x nu

    // sense
    mjData* d = mj_makeData(m);
    set_data_by_sensor(sensordata, m, d);

    VectorXd start_state;
    get_state_from_data(m, d, start_state);

    /* obtain samples */
    std::vector<MatrixXd> u_samples, x_samples;
    // eps_x0 = x0 - x_n0. 
    // x0 is from observation

    sample(start_state, x_samples, u_samples);

    // TODO: set one sample to be u=0
    std::vector<double> x_bound_costs;
    get_x_bound_cost(x_samples, x_bound_costs);
    // mjData* new_d = mj_makeData(m);

    double cost = 0;
    std::vector<double> sample_costs;
    sample_costs.resize(N);

    // std::cout << "u_samples: " << std::endl;
    // for (int i=0; i<u_samples.size(); i++)
    // {
    //     std::cout << u_samples[i] << std::endl;
    // }

    // std::cout << "x_samples: " << std::endl;
    // for (int i=0; i<x_samples.size(); i++)
    // {
    //     std::cout << x_samples[i] << std::endl;
    // }

    // omp_set_num_thread

    std::vector<mjData*> new_ds;
    for (int i=0; i<N; i++)
    {
        new_ds.push_back(mj_makeData(m));
    }


    omp_set_num_threads(10);

    #pragma omp parallel for
    for (int i=0; i<N; i++)
    {
        sample_costs[i] = 0;
        sample_costs[i] += bound_cost_scale * x_bound_costs[i];
        // reset simulation
        mj_copyData(new_ds[i], m, d);
        // mj_resetData(m, new_d);

        for (int j=0; j<H; j++)
        {
            /* compute cost */
            // TODO: add cost on disturbance

            // problem-specific cost
            double cost_h = get_cost(m, new_ds[i]);
            sample_costs[i] += cost_h;

            int n_steps = ceil(dt / m->opt.timestep);

            int nominal_idx = std::min(nominal_start_idx+j, static_cast<int>(nominal_u.rows())-1);
            nominal_idx = std::max(nominal_idx, 0);

            int state_nominal_idx = std::min(nominal_start_idx+j+1, static_cast<int>(nominal_x.rows())-1);
            state_nominal_idx = std::max(state_nominal_idx, 0);

            for (int step=0; step<n_steps; step++)
            {
                // set the control according to the position and velocity
                for (int k=0; k<x_samples[i].cols(); k++)
                {

                    // new_ds[i]->act[pos_act_indices[k]] = x_samples[i](j+1,k) + nominal_x(state_nominal_idx,k);
                    new_ds[i]->ctrl[vel_ctrl_indices[k]] = u_samples[i](j,k) + nominal_u(nominal_idx,k);
                }
                // step
                mj_step(m, new_ds[i]);
            }
        }
        double terminal_cost = get_terminal_cost(m, new_ds[i]);
        sample_costs[i] += terminal_cost;
    }


    // parallel for:
    //   u = nominal_u + eps[i]
    //   system.reset(x)
    //   for h = 0 .. H:
    //      cost(x,u)
    //      system.propagate(x,u[i,h]) -> x
    //   terminal_cost()


    std::cout << "before updating distribution..." << std::endl;
    /* update distributions according to the cost */
    double beta = 0.1;
    // weight: w[i] = exp{(-1/beta) * (sample_costs[i])}
    std::vector<double> weights;
    double sum = 0;
    weights.resize(N);
    // for (int i=0; i<sample_costs.size(); i++)
    // {
    //     std::cout << sample_costs[i] << ", ";
    // }
    // std::cout << std::endl;


    for (int i=0; i<sample_costs.size(); i++)
    {
        weights[i] = exp(-1/beta*sample_costs[i]);
        sum += weights[i];
    }
    // mu: weighted average of sampled controls
    for (int h = 0; h<H; h++)
    {
        VectorXd u_weighted_sum(u_ll.size());
        u_weighted_sum.setZero();
        for (int i=0; i<N; i++)
        {
            u_weighted_sum += u_samples[i].row(h) * weights[i];
        }
        u_weighted_sum = u_weighted_sum / sum;
        mu[h] = u_weighted_sum;
    }

    // std::cout << "updated mu: " << std::endl;
    // for (int i=0; i<mu.size(); i++)
    // {
    //     std::cout << mu[i].transpose() << std::endl;
    // }

    // x_mu: weighted average of sampled states
    VectorXd x_mu_1(x_ll.size());
    x_mu_1.setZero();

    VectorXd x_weighted_sum(x_ll.size());
    x_weighted_sum.setZero();
    for (int i=0; i<N; i++)
    {
        // we are feeding the next state as the target state to track
        x_weighted_sum += x_samples[i].row(1) * weights[i];
    }
    x_weighted_sum = x_weighted_sum / sum;
    x_mu_1 = x_weighted_sum;

    // control: act | ctrl

    int nominal_idx = nominal_start_idx+0;
    int x_nominal_idx = min(0, nominal_start_idx+1);
    control.resize(pos_act_indices.size() + vel_ctrl_indices.size());
    control.head(x_mu_1.size()) = x_mu_1 + nominal_x.row(x_nominal_idx).transpose();
    control.tail(mu[0].size()) = mu[0] + nominal_u.row(nominal_idx).transpose();

    // for (int i=0; i<mu[0].size(); i++)
    // {
    //     d->act[pos_act_indices[i]] = x_mu_1[i] + nominal_x(nominal_idx,i);
    //     d->ctrl[vel_ctrl_indices[i]] = mu[0][i] + nominal_u(nominal_idx,i);
    // }

    // shift the mu
    mu.pop_front();
    VectorXd new_mu(u_ll.size());
    new_mu.setZero();
    mu.push_back(new_mu);
    if (nominal_start_idx < nominal_x.rows()-1)
    {
        nominal_start_idx += 1;
    }

    // handle data
    delete d;
    // delete new_d;
    for (int i=0; i<N; i++)
    {
        delete new_ds[i];
    }

}

void MujocoMPPIControllerIntvel::get_x_bound_cost(const std::vector<MatrixXd>& x_samples,
                                                  std::vector<double>& x_bound_costs)
{
    x_bound_costs.resize(x_samples.size());
    for (int i=0; i<x_samples.size(); i++)
    {
        x_bound_costs[i] = 0;
        for (int j=0; j<x_samples[i].rows(); j++)
        {
            int nominal_idx = std::min(nominal_start_idx+j, static_cast<int>(nominal_x.rows())-1);
            nominal_idx = std::max(nominal_idx, 0);
            for (int k=0; k<x_samples[i].cols(); k++)
            {
                double x = x_samples[i](j,k) + nominal_x(nominal_idx,k);
                if (x < x_ll[k])
                {
                    // quad cost
                    x_bound_costs[i] += (x - x_ll[k])*(x - x_ll[k]);
                }
                if (x > x_ul[k])
                {
                    // quad cost
                    x_bound_costs[i] += (x - x_ul[k])*(x - x_ul[k]);
                }
            }
        }
    }
}
