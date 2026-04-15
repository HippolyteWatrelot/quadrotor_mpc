#ifndef CASADI_MODEL_H_INCLUDED
#define CASADI_MODEL_H_INCLUDED

#include <map>
#include <vector>
#include <cmath>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <assert.h>
#include <ros/ros.h>
#include <ros/node_handle.h>

#include <casadi/casadi.hpp>


class casadi_model
{
public:
    casadi_model(bool gt, bool w);
    ~casadi_model();
    std::vector<int> T;
    std::vector<int> control_dim;
    casadi::SX A, B, C, D, K;
    casadi::SX Q, R;
    casadi::SX NQ, NR;
    std::vector<double> Kvec;
    Eigen::MatrixXd Gamma, H, Phi;
    void cost_matrices();
    void N_cost_matrices();
    void build_model();
    /*static casadi::SX extract_U(casadi::SX Z, casadi::SX Uact, int nx, int nu);
    static casadi::SX f_memory(casadi::SX u, int nu, double r[]);*/

protected:
    void init_parameters();
    void build_ABCD();
    //static double cost(const Eigen::VectorXd& c, const Eigen::VectorXd& x, const Eigen::VectorXd& xref, const Eigen::VectorXd& uref);

private:
    bool ground_truth;
    bool wind;
    int N;
    int nx=12; 
    int nu=8;
    std::vector<double> sol_prev;
    casadi::Function solver;
    std::map<std::string, casadi::DM> args;
    double delta_t;
    double dt=0.01;
    double lxy_kp, lxy_ki, lxy_kd;
    double lz_kp, lz_ki, lz_kd;
    double az_kp, az_ki, az_kd;
    double lxy_tau, lz_tau, az_tau;
    ros::NodeHandle _nh;
};

#endif // CASADI_MODEL_H_INCLUDED
