#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <map>
#include <string>

#include "quadrotor_mpc/casadi_model.h"
#include <casadi/casadi.hpp>
#include <limits>


using namespace casadi;



casadi_model::casadi_model(bool gt, bool w)
{
    ground_truth = gt; 
    wind = w; 
    init_parameters();
}

casadi_model::~casadi_model() {}


void casadi_model::build_ABCD()
{
    assert(T.size() == 4);
    
    int n(std::accumulate(T.begin(), T.end(), 0));
    //int p(std::accumulate(control_dim.begin(), control_dim.end(), 0));
    Eigen::MatrixXd _A(n, n), _B(n, 8);
    _A.setZero(n, n); _B.setZero(n, 8);
    double kp[4] = {lxy_kp, lxy_kp, lz_kp, az_kp};
    double ki[4] = {lxy_ki, lxy_ki, lz_ki, az_ki};
    double kd[4] = {lxy_kd, lxy_kd, lz_kd, az_kd};
    double r[4] = {delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lz_tau), delta_t / (delta_t + az_tau)};
    int dim_T = 0;
    int dim_C = 0;
    for (int i(0); i<4; i++)
    {
        Eigen::MatrixXd a(T[i], T[i]), b(1, 2);
        double gamma(kp[i] + ki[i]*delta_t + kd[i]/delta_t);
        if (T[i] == 6)
        {
            a << 1, 0, delta_t, 0, delta_t*delta_t/2,
                 0, 1, 0, delta_t, 0,
                 0, 0, 1, 0, delta_t,
                 0, 0, 0, 0, 1, 0,
                 0, kd[i]/delta_t, -gamma, kd[i], -kd[i];
        }
        else
        {
            a << 1, delta_t, delta_t*delta_t/2,
                 0, 1, delta_t,
                 0, -gamma, 0;
        }
        b << (1-r[i]) * gamma - kd[i]/delta_t, r[i] * gamma;
        for (int k(0); k<T[i]; k++)
            for (int m(0); m<T[i]; m++) {_A(dim_T + k, dim_T + m) = a(k, m);}
        for (int k(0); k<2; k++) {_B(dim_T + T[i] - 1, dim_C + k) = b(0, k);}
        dim_T += T[i];
        dim_C += 2;
    }
    A = DM::zeros(_A.rows(), _A.cols());
    B = DM::zeros(_B.rows(), _B.cols());
    for (int i(0); i<_A.rows(); i++)
    {
        for (int j(0); j<_A.cols(); j++)
            A(i, j) = _A(i, j);
        for (int j(0); j<_B.cols(); j++)
            B(i, j) = _B(i, j);
    }
    if (!ground_truth) {Eigen::MatrixXd c(n, n);};
    if (wind) {};
    // ...
}


void casadi_model::init_parameters() 
{
    _nh.getParam("controller/twist/linear/xy/k_p", lxy_kp);
    _nh.getParam("controller/twist/linear/xy/k_i", lxy_ki);
    _nh.getParam("controller/twist/linear/xy/k_d", lxy_kd);
    _nh.getParam("controller/twist/linear/xy/time_constant", lxy_tau);
    
    _nh.getParam("controller/twist/linear/z/k_p", lz_kp);
    _nh.getParam("controller/twist/linear/z/k_i", lz_ki);
    _nh.getParam("controller/twist/linear/z/k_d", lz_kd);
    _nh.getParam("controller/twist/linear/z/time_constant", lz_tau);
    
    _nh.getParam("controller/twist/angular/z/k_p", az_kp);
    _nh.getParam("controller/twist/angular/z/k_i", az_ki);
    _nh.getParam("controller/twist/angular/z/k_d", az_kd);
    _nh.getParam("controller/twist/angular/z/time_constant", az_tau);
    
    if (lxy_kd == 0.0) {T.push_back(3);} else {T.push_back(5);};
    if (lxy_kd == 0.0) {T.push_back(3);} else {T.push_back(5);};
    if (lz_kd == 0.0) {T.push_back(3);} else {T.push_back(5);};
    if (az_kd == 0.0) {T.push_back(3);} else {T.push_back(5);};
    
    //if (lxy_kd == 0.0) {T.push_back(3); control_dim.push_back(1);} else {T.push_back(5); control_dim.push_back(2);};
    //if (lxy_kd == 0.0) {T.push_back(3); control_dim.push_back(1);} else {T.push_back(5); control_dim.push_back(2);};
    //if (lz_kd == 0.0) {T.push_back(3); control_dim.push_back(1);} else {T.push_back(5); control_dim.push_back(2);};
    //if (az_kd == 0.0) {T.push_back(3); control_dim.push_back(1);} else {T.push_back(5); control_dim.push_back(2);};
    
    _nh.getParam("N", N);
    _nh.getParam("delta_t", delta_t);
    
    _nh.getParam("K", Kvec);
    K = DM::zeros(nu, nx);
    for (int i(0); i<nu; i++) 
    {
        for (int j(0); j<nx; j++)
        {
            K(i, j) = Kvec[i*nx + j];
            std::cout << K(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    build_ABCD();
    cost_matrices();
    N_cost_matrices();
}


void casadi_model::cost_matrices()
{
    std::vector<double> Qvec;
    std::vector<double> Rvec;
    _nh.getParam("Q", Qvec);
    _nh.getParam("R", Rvec);
    Q = SX::zeros(nx, nx);
    R = SX::zeros(nu, nu);
    for (int i(0); i<nx; i++) {Q(i, i) = Qvec[i];}
    for (int i(0); i<nu; i++) {R(i, i) = Rvec[i];}
}


void casadi_model::N_cost_matrices()
{
    std::vector<double> Qvec;
    std::vector<double> Rvec;
    _nh.getParam("Q", Qvec);
    _nh.getParam("R", Rvec);
    NQ = SX::zeros(nx*N, nx*N);
    NR = SX::zeros(nu*N, nu*N);
    for (int i(0); i<nx*N; i++) {NQ(i, i) = Qvec[i];}
    for (int i(0); i<nu*N; i++) {NR(i, i) = Rvec[i];}
}


/*
SX extract_U(SX Z, SX Uact, int nx, int nu)
{
    SX Uf = SX::zeros(nu);
    for (int i(0); i<nu/2; i++)
    {
        Uf(2*i) = Z(nx + i);
        Uf(2*i+1) = Uact(i);
    }
    return Uf;
}

SX f_memory(SX u, int nu, double r[])
{
    SX res = SX::zeros(nu/2);
    for (int i(0); i<nu/2; i++)
        res(i) = (1-r[i]) * u(2*i) + r[i] * u(2*i+1);
    return res;
}
*/



void casadi_model::build_model()
{
    double kp[4] = {lxy_kp, lxy_kp, lz_kp, az_kp};
    double ki[4] = {lxy_ki, lxy_ki, lz_ki, az_ki};
    double kd[4] = {lxy_kd, lxy_kd, lz_kd, az_kd};
    double r[4] = {delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lz_tau), delta_t / (delta_t + az_tau)};

    ROS_INFO("Building System...");
    SX Z = SX::sym("Z", nx + nu/2);
    SX Uactive = SX::sym("U", nu/2);
    
    // Extracting state
    SX X = Z(Slice(0, nx));
    
    // Extracting Full control vector
    SX U = SX::zeros(nu);
    for (int i(0); i<nu/2; i++)
    {
        U(2*i) = Z(nx + i);
        U(2*i+1) = Uactive(i);
    }
    
    // Dynamical system
    SX X_next = mtimes(A, X) + mtimes(B, U);
    
    // New passive control
    SX U_passive_next = SX::zeros(nu/2);
    for (int i(0); i<nu/2; i++)
        U_passive_next(i) = (1-r[i]) * U(2*i) + r[i] * U(2*i+1);
    
    // New Extended state
    SX Z_next = vertcat(X_next, U_passive_next);
    
    // Extended system
    Function dyn_fun = Function("dyn_fun", {Z, Uactive}, {Z_next});
    
    CodeGenerator cg("drone_dynamics");
    cg.add(dyn_fun);
    cg.generate();
    
    ROS_INFO("model_dynamics.c/h generated");
}
