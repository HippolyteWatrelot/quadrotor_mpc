#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <map>
#include <string>

#include "std_msgs/Int16.h"
#include "std_msgs/Float32MultiArray.h"

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/Wrench.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "geometry_msgs/Twist.h"

#include "quadrotor_mpc/MPC.h"

#include <qpOASES.hpp>
#include <limits>


using namespace qpOASES;



MPC::MPC(int Horizon, bool gt, bool w)
{
    ground_truth = gt; 
    wind = w; 
    init_parameters();
    N = Horizon;
}

MPC::~MPC() {}


void MPC::build_ABCD()
{
    assert(T.size() == 4);
    
    int n(std::accumulate(T.begin(), T.end(), 0));
    //int p(std::accumulate(control_dim.begin(), control_dim.end(), 0));
    Eigen::MatrixXd _A(n, n), _B(n, 4);
    _A.setZero(n, n); _B.setZero(n, 4);
    double kp[4] = {lxy_kp, lxy_kp, lz_kp, az_kp};
    double ki[4] = {lxy_ki, lxy_ki, lz_ki, az_ki};
    double kd[4] = {lxy_kd, lxy_kd, lz_kd, az_kd};
    double r[4] = {delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lz_tau), delta_t / (delta_t + az_tau)};
    int dim_T = 0;
    int dim_C = 0;
    for (int i(0); i<4; i++)
    {
        Eigen::MatrixXd a(T[i], T[i]), b(T[i], 1);
        double gamma(kp[i] + ki[i]*delta_t + kd[i]/delta_t);
        if (T[i] == 6)
        {
            a << 1, 0, delta_t, 0, delta_t*delta_t/2, 0,
                 0, 1, 0, delta_t, 0, 0,
                 0, 0, 1, 0, delta_t, 0,
                 0, 0, 0, 0, 1, 0,
                 0, kd[i]/delta_t, -gamma, kd[i], -kd[i], (1-r[i])*gamma -kd[i]/delta_t,
                 0, 0, 0, 0, 0, 1-r[i];
            b << 0, 0, 0, 0, r[i] * gamma, r[i];
        }
        else
        {
            a << 1, delta_t, delta_t*delta_t/2, 0,
                 0, 1, delta_t, 0,
                 0, -gamma, 0, (1-r[i])*gamma,
                 0, 0, 0, 1-r[i];
            b << 0, 0, r[i] * gamma, r[i];
        }
        for (int k(0); k<T[i]; k++)
            for (int m(0); m<T[i]; m++) {_A(dim_T + k, dim_T + m) = a(k, m);}
        for (int k(2); k<4; k++) 
            for (int m(0); m<T[i]; m++) {_B(dim_T + k, dim_C) = b(k, 0);}
        dim_T += T[i];
        dim_C += 1;
    }
    A = _A; B= _B;
    if (!ground_truth) {Eigen::MatrixXd c(n, n);};
    if (wind) {};
    // ...
}


void MPC::init_parameters() 
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
    
    if (lxy_kd == 0.0) {T.push_back(4);} else {T.push_back(6);};
    if (lxy_kd == 0.0) {T.push_back(4);} else {T.push_back(6);};
    if (lz_kd == 0.0) {T.push_back(4);} else {T.push_back(6);};
    if (az_kd == 0.0) {T.push_back(4);} else {T.push_back(6);};
    
    //if (lxy_kd == 0.0) {T.push_back(3); control_dim.push_back(1);} else {T.push_back(5); control_dim.push_back(2);};
    //if (lxy_kd == 0.0) {T.push_back(3); control_dim.push_back(1);} else {T.push_back(5); control_dim.push_back(2);};
    //if (lz_kd == 0.0) {T.push_back(3); control_dim.push_back(1);} else {T.push_back(5); control_dim.push_back(2);};
    //if (az_kd == 0.0) {T.push_back(3); control_dim.push_back(1);} else {T.push_back(5); control_dim.push_back(2);};
    
    _nh.getParam("N", N);
    _nh.getParam("delta_t", delta_t);
    
    build_ABCD();
    get_K();
}


void MPC::get_K()
{
    XmlRpc::XmlRpcValue Kvec;
    _nh.getParam("K", Kvec);
    assert(Kvec.getType() == XmlRpc::XmlRpcValue::TypeArray);
    Eigen::MatrixXd k(16, 4);
    for (int i(0); i<16; i++)
    {
        for (int j(0); j<4; j++)
            k(i, j) = Kvec[i][j];
    }
    K = k;
}


void MPC::get_mats()
{
    XmlRpc::XmlRpcValue matX, matC;
    _nh.getParam("Mx", matX);
    _nh.getParam("Mc", matC);
    assert(matX.getType() == XmlRpc::XmlRpcValue::TypeArray);
    assert(matC.getType() == XmlRpc::XmlRpcValue::TypeArray);
    Eigen::MatrixXd mx(N*16, 16);
    Eigen::MatrixXd mc(N*16, N*4);
    for (int i(0); i<N*16; i++)
    {
        for (int j(0); j<16; j++)
            mx(i, j) = matX[i][j];
        for (int j(0); j<N*4; j++)
            mc(i, j) = matC[i][j];
    }
    Mx = mx;
    Mc = mc;
}


void MPC::cost_matrices()
{
    XmlRpc::XmlRpcValue Qvec, Rvec;
    _nh.getParam("Q", Qvec);
    _nh.getParam("R", Rvec);
    assert(Qvec.getType() == XmlRpc::XmlRpcValue::TypeArray);
    assert(Rvec.getType() == XmlRpc::XmlRpcValue::TypeArray);
    Eigen::MatrixXd q(N*16, N*16), r(N*4, N*4);
    for (int i(0); i<16; i++)
    {
        for (int j(0); j<16; j++)
            q(i, j) = Qvec[i][j];
    }
    for (int i(0); i<4; i++)
    {
        for (int j(0); j<4; j++)
            r(i, j) = Rvec[i][j];
    }
    Q = q;
    R = r;
}


void MPC::N_cost_matrices()
{
    XmlRpc::XmlRpcValue Qvec, Rvec;
    _nh.getParam("Q", Qvec);
    _nh.getParam("R", Rvec);
    assert(Qvec.getType() == XmlRpc::XmlRpcValue::TypeArray);
    assert(Rvec.getType() == XmlRpc::XmlRpcValue::TypeArray);
    Eigen::MatrixXd q(N*16, N*16), r(N*4, N*4);
    for (int i(0); i<N*16; i++)
    {
        for (int j(0); j<N*16; j++)
            q(i, j) = Qvec[i][j];
    }
    for (int i(0); i<N*4; i++)
    {
        for (int j(0); j<N*4; j++)
            r(i, j) = Rvec[i][j];
    }
    NQ = q;
    NR = r;
}


Eigen::MatrixXd MPC::matrix_power(Eigen::MatrixXd M, int n)
{
    Eigen::MatrixXd S(M);
    for (int i(0); i<n-1; i++)
        S = M * S;
    return S;
}


void MPC::constraints_matrices()
{
    
}


Eigen::MatrixXd MPC::Riccati_solve(int max_iter, double tol)
{
    Eigen::MatrixXd P = Q;
    for (int i = 0; i < max_iter; ++i) {
        Eigen::MatrixXd K = (R + B.transpose() * P * B).inverse() * (B.transpose() * P * A);
        Eigen::MatrixXd P_next = Q + A.transpose() * P * A - A.transpose() * P * B * K;

        if ((P_next - P).norm() < tol) {
            std::cout << "Converged after " << i << " iterations." << std::endl;
            K = (R + B.transpose() * P_next * B).inverse() * (B.transpose() * P_next * A);
            return K;
        }
        P = P_next;
    }
    std::cout << "Did not converge." << std::endl;
    return K;
}


void MPC::optimize(const Eigen::VectorXd& x, const Eigen::VectorXd& xref, const Eigen::VectorXd& uref)
{
    
}


void MPC::run()
{
    USING_NAMESPACE_QPOASES
    
    Eigen::VectorXd x0;
    x0 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(N * 16, 16);
    Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(N * 16, N * 4);                      // Horizon Matrices
    
    for (int i = 0; i < N; i++) {
        //Phi.block(i * 12, 0, 12, 12) = matrix_power(A, i+1);
        Phi.block<16, 16>(i * 16, 0) = matrix_power(A, i+1);
        for (int j = 0; j <= i; j++) {
            //Gamma.block(i * 12, j * 8, 12, 8) = matrix_power(A, i-j) * B;
            Gamma.block<16, 4>(i * 16, j * 4) = matrix_power(A, i-j) * B;
        }
    }
    
    Eigen::MatrixXd H = 2 * Gamma.transpose() * NQ * Gamma + NR;
    Eigen::VectorXd g = 2 * Gamma.transpose() * NQ * Phi * x0;
    
    real_t H_qp[N * nu * N * nu];
    real_t g_qp[N * nu];
    for (int i = 0; i < N * nu; i++) {
        g_qp[i] = g(i);
        for (int j = 0; j < N * nu; j++) {
            H_qp[i * N * nu + j] = H(i, j);
        }
    }
    
    /*real_t u_lb[N * 4];
    real_t u_ub[N * 4];
    for (int i = 1; i < N * 4; i+2) {
        u_lb[i] = umin[(i-1)/2];
        u_ub[i] = umax[(i-1)/2];
    }*/

    // QP Solver
    /*QProblem qp(N * nu, 0);  // Pas de contraintes générales ici
    Options options;
    options.setToMPC();
    qp.setOptions(options);

    int nWSR = 20;
    qp.init(H_qp, g_qp, nullptr, u_lb, u_ub, nullptr, nullptr, nWSR);

    real_t u_opt[N * nu];
    qp.getPrimalSolution(u_opt);

    std::cout << "Commande optimale u₀ (avec contraintes) : " << u_opt[0] << std::endl;*/
}
