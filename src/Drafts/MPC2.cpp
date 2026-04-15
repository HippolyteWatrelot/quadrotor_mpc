#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <map>
#include <string>

#include "quadrotor_mpc/MPC.h"

#include <qpOASES.hpp>
#include <limits>


using namespace qpOASES;



MPC::MPC(bool gt, bool w)
{
    ground_truth = gt; 
    wind = w; 
    init_parameters();
}

MPC::~MPC() {}


void MPC::build_ABCD()
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
    A = _A; B= _B;
    if (!ground_truth) {Eigen::MatrixXd c(n, n);};
    if (wind) {};
    // ...
}


void MPC::pose_callback(const geometry_msgs::PoseStamped::ConstPtr& pose)
{
    double prev_pos_x = current_pose(0);
    double prev_pos_y = current_pose(1);
    double prev_pos_z = current_pose(2);
    current_pose(0) = pose->pose.position.x;
    current_pose(1) = pose->pose.position.y;
    current_pose(2) = pose->pose.position.z;
    current_ori(0) = pose->pose.orientation.w;
    current_ori(1) = pose->pose.orientation.x;
    current_ori(2) = pose->pose.orientation.y;
    current_ori(3) = pose->pose.orientation.z;
    double prev_vel_x = current_vel(0);
    double prev_vel_y = current_vel(1);
    double prev_vel_z = current_vel(2);
    current_vel(0) = (current_pose(0) - prev_pos_x) / dt;
    current_vel(1) = (current_pose(1) - prev_pos_y) / dt;
    current_vel(2) = (current_pose(2) - prev_pos_z) / dt;
    current_acc(0) = (current_vel(0) - prev_vel_x) / dt;
    current_acc(1) = (current_vel(1) - prev_vel_y) / dt;
    current_acc(2) = (current_vel(2) - prev_vel_z) / dt;
}


void MPC::euler_callback(const geometry_msgs::Vector3Stamped::ConstPtr& vec)
{
    double prev_yaw = current_euler(2);
    double prev_yaw_vel = current_yaw_vel;
    current_euler(0)  = vec->vector.x;
    current_euler(1) = vec->vector.y;
    current_euler(2)   = vec->vector.z;
    current_yaw_vel = (current_euler(2) - prev_yaw) / dt;
    current_yaw_acc = (current_yaw_vel - prev_yaw_vel) / dt;
}


void MPC::command_callback(const geometry_msgs::Twist::ConstPtr& twist)
{
    current_command(0) = twist->linear.x;
    current_command(1) = twist->linear.y;
    current_command(2) = twist->linear.z;
    current_command(3) = twist->angular.z;
}


void MPC::traj_length_callback(const std_msgs::Int16::ConstPtr& msg)
{
    trajL = msg->data;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(12*trajL);
    xref = x;
    Eigen::VectorXd u = Eigen::VectorXd::Zero(8*trajL);
    uref = u;
    ROS_INFO("traj length: %d", trajL);
}


void MPC::traj_points_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        xref(12*i) = t->data[3*i];
        xref(12*i+4) = t->data[3*i+1];
        xref(12*i+8) = t->data[3*i+2];
    }
}


void MPC::traj_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        xref(12*i+1) = t->data[3*i];
        xref(12*i+5) = t->data[3*i+1];
        xref(12*i+9) = t->data[3*i+2];
    }
}


void MPC::traj_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        xref(12*i+2) = t->data[3*i];
        xref(12*i+6) = t->data[3*i+1];
        xref(12*i+10) = t->data[3*i+2];
    }
}


void MPC::traj_yaws_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        xref(12*i+3) = t->data[i];
}


void MPC::traj_yaws_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        xref(12*i+7) = t->data[i];
}


void MPC::traj_yaws_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        xref(12*i+11) = t->data[i];
}


void MPC::timestep_callback(const std_msgs::Float32::ConstPtr& dt)
{
    delta_t = dt->data;
    ROS_INFO("time step: %f", delta_t);
    build_ABCD();
    int n(std::accumulate(T.begin(), T.end(), 0));
    std::cout << std::endl << std::endl;;
    std::cout << "A: " << std::endl;
    for (int i(0); i<n; i++)
    {
        for (int j(0); j<n; j++)
            std::cout << A(i, j) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;;
    std::cout << "B: " << std::endl;
    for (int i(0); i<n; i++)
    {
        for (int j(0); j<8; j++)
            std::cout << B(i, j) << " ";
        std::cout << std::endl;
    }
}


void MPC::ui_command_callback(const std_msgs::Int16::ConstPtr& msg)
{
    if (msg->data==13) {run_trajectory();}
    else if (msg->data==14) 
    {
        ROS_INFO_STREAM("position: " << current_pose(0) << " " << current_pose(1) << " " << current_pose(2));
        ROS_INFO_STREAM("roll: " << current_euler(0) << ", pitch: " << current_euler(1) << ", yaw: " << current_euler(2));
    }
    else if (msg->data==15)
    {
        try
        {
            assert(trajL >= 1);
            std::cout << "length " << trajL << " REGISTERED TRAJECTORY:" << std::endl;
            for (int i(0); i<trajL; i++)
            {
                int n(12*i);
                std::cout << xref[n] << " "<< xref[n+1] <<" "<< xref[n+2] <<" "<< xref[n+3] <<" "<< xref[n+4] <<" "<< xref[n+5] <<" "<< xref[n+6] <<" "<< xref[n+7] <<" "<< xref[n+8] <<" "<< xref[n+9] <<" "<< xref[n+10] <<" "<< xref[n+11] << std::endl;
            }
        }
        catch(...) {ROS_WARN("NO TRAJ RECORDED");}
    }
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
    _nh.getParam("umin", umin);
    _nh.getParam("umax", umax);
    
    _nh.getParam("K", K);
    /*for (int i(0); i<nx; i++) 
    {
        for (int j(0); j<nu; j++)
            K(i, j) = K[i*nu + j];
    }*/
    
    Gamma = Eigen::MatrixXd::Zero(N*nx, N*nu);
    std::vector<double> Gamma_vec;
    _nh.getParam("Gamma", Gamma_vec);
    for (int i(0); i<N*nx; i++)
    {
        for (int j(0); j<N*nu; j++)
            Gamma(i, j) = Gamma_vec[i*N*nu + j];
    }
    printMatrix("Gamma", Gamma, N*nx, N*nu);
    
    H = Eigen::MatrixXd::Zero(N*nu, N*nu);
    std::vector<double> Hvec;
    _nh.getParam("H", Hvec);
    for (int i(0); i<N*nu; i++)
    {
        for (int j(0); j<N*nu; j++)
            H(i, j) = Hvec[i*N*nu + j];
    }
    printMatrix("H", H, N*nu, N*nu);
    
    Phi = Eigen::MatrixXd::Zero(N*nx, nx);
    std::vector<double> Pvec;
    _nh.getParam("phi", Pvec);
    for (int i(0); i<N*nx; i++)
    {
        for (int j(0); j<nx; j++)
            Phi(i, j) = Pvec[i*nx + j];
    }
    printMatrix("Phi", Phi, N*nx, nx);
    
    current_pose << 0, 0, 0;
    current_euler << 0, 0, 0;
    current_ori << 0, 0, 0, 0;
    xref = Eigen::VectorXd::Zero(nx*N);
    uref = Eigen::VectorXd::Zero(nu*N);
    
    pose_subscriber    = _nh.subscribe<geometry_msgs::PoseStamped>("/ground_truth_to_tf/pose", 1, &MPC::pose_callback, this);
    euler_subscriber   = _nh.subscribe<geometry_msgs::Vector3Stamped>("/ground_truth_to_tf/euler", 1, &MPC::euler_callback, this);
    command_subscriber = _nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 1, &MPC::command_callback, this);
    
    traj_length_subscriber = _nh.subscribe<std_msgs::Int16>("/trajectory_length", 1, &MPC::traj_length_callback, this);
    traj_points_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_points", 1, &MPC::traj_points_callback, this);
    traj_speeds_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_speeds", 1, &MPC::traj_speeds_callback, this);
    traj_accs_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_accs", 1, &MPC::traj_accs_callback, this);
    
    yaws_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_yaws", 1, &MPC::traj_yaws_callback, this);
    yaws_speeds_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_yaws_speeds", 1, &MPC::traj_yaws_speeds_callback, this);
    yaws_accs_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_yaws_accs", 1, &MPC::traj_yaws_accs_callback, this);
    
    timestep_subscriber = _nh.subscribe<std_msgs::Float32>("/timestep", 1, &MPC::timestep_callback, this);
    ui_command_subscriber = _nh.subscribe<std_msgs::Int16>("/command", 1, &MPC::ui_command_callback, this);
    
    exe_pub = _nh.advertise<geometry_msgs::Twist>("/cmd_vel", 500, true);
    
    build_ABCD();
    cost_matrices();
    N_cost_matrices();
}


Eigen::Vector3d MPC::toBody(const Eigen::Vector3d& space)
{
    const double& w = current_ori(0);
    const double& x = current_ori(1);
    const double& y = current_ori(2);
    const double& z = current_ori(3);
    Eigen::Vector3d body;
    body(0) = (w*w+x*x-y*y-z*z) * space(0) + (2.*x*y + 2.*w*z) * space(1) + (2.*x*z - 2.*w*y) * space(2);
    body(1) = (2.*x*y - 2.*w*z) * space(0) + (w*w-x*x+y*y-z*z) * space(1) + (2.*y*z + 2.*w*x) * space(2);
    body(2) = (2.*x*z + 2.*w*y) * space(0) + (2.*y*z - 2.*w*x) * space(1) + (w*w-x*x-y*y+z*z) * space(2);
    return body;
}


Eigen::Vector3d MPC::fromBody(const Eigen::Vector3d& body)
{
    const double& w = current_ori(0);
    const double& x = current_ori(1);
    const double& y = current_ori(2);
    const double& z = current_ori(3);
    Eigen::Vector3d space;
    space(0) = (w*w+x*x-y*y-z*z) * body(0) + (2.*x*y - 2.*w*z) * body(1) + (2.*x*z + 2.*w*y) * body(2);
    space(1) = (2.*x*y + 2.*w*z) * body(0) + (w*w-x*x+y*y-z*z) * body(1) + (2.*y*z - 2.*w*x) * body(2);
    space(2) = (2.*x*z - 2.*w*y) * body(0) + (2.*y*z + 2.*w*x) * body(1) + (w*w-x*x-y*y+z*z) * body(2);
    return space;
}


void MPC::get_mats()
{
    XmlRpc::XmlRpcValue matX, matC;
    _nh.getParam("Mx", matX);
    _nh.getParam("Mc", matC);
    assert(matX.getType() == XmlRpc::XmlRpcValue::TypeArray);
    assert(matC.getType() == XmlRpc::XmlRpcValue::TypeArray);
    Eigen::MatrixXd mx(N*12, 12);
    Eigen::MatrixXd mc(N*12, N*8);
    for (int i(0); i<N*12; i++)
    {
        for (int j(0); j<12; j++)
            mx(i, j) = matX[i][j];
        for (int j(0); j<N*8; j++)
            mc(i, j) = matC[i][j];
    }
    Mx = mx;
    Mc = mc;
}


void MPC::cost_matrices()
{
    //XmlRpc::XmlRpcValue Qvec, Rvec;
    std::vector<double> Qvec;
    std::vector<double> Rvec;
    _nh.getParam("Q", Qvec);
    _nh.getParam("R", Rvec);
    //assert(Qvec.getType() == XmlRpc::XmlRpcValue::TypeArray);
    //assert(Rvec.getType() == XmlRpc::XmlRpcValue::TypeArray);
    //Eigen::MatrixXd q(12, 12), r(8, 8);
    Q = Eigen::MatrixXd::Zero(nx, nx);
    R = Eigen::MatrixXd::Zero(nu, nu);
    for (int i(0); i<nx; i++) {Q(i, i) = Qvec[i];}
    for (int i(0); i<nu; i++) {R(i, i) = Rvec[i];}
    //Q = q;
    //R = r;
}


void MPC::N_cost_matrices()
{
    //XmlRpc::XmlRpcValue Qvec, Rvec;
    std::vector<double> Qvec;
    std::vector<double> Rvec;
    _nh.getParam("Q", Qvec);
    _nh.getParam("R", Rvec);
    //assert(Qvec.getType() == XmlRpc::XmlRpcValue::TypeArray);
    //assert(Rvec.getType() == XmlRpc::XmlRpcValue::TypeArray);
    NQ = Eigen::MatrixXd::Zero(nx*N, nx*N);
    NR = Eigen::MatrixXd::Zero(nu*N, nu*N);
    for (int i(0); i<nx*N; i++) {NQ(i, i) = Qvec[i];}
    for (int i(0); i<nu*N; i++) {NR(i, i) = Rvec[i];}
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


void MPC::printMatrix(const std::string& name, Eigen::MatrixXd mat, int rows, int cols) 
{
    std::cout << name << " = \n";
    for (int i = 0; i < rows; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < cols; ++j) {
            std::cout << mat(i, j) << " ";  // Col-major!
        }
        std::cout << "]\n";
    }
}


void MPC::run_trajectory()
{
    USING_NAMESPACE_QPOASES
    
    double kp[4] = {lxy_kp, lxy_kp, lz_kp, az_kp};
    double ki[4] = {lxy_ki, lxy_ki, lz_ki, az_ki};
    double kd[4] = {lxy_kd, lxy_kd, lz_kd, az_kd};
    double r[4]  = {delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lz_tau), delta_t / (delta_t + az_tau)};
    
    Eigen::VectorXd x0 = Eigen::Vector3d::Zero(nx);
    //x0 << xref.head(12);
    x0(0) = current_pose(0); 
    x0(1) = current_pose(1); 
    x0(2) = current_pose(2); 
    x0(3) = current_euler(2);
    
    /*Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(N*nx, nx);                      // Horizon Matrices
    for (int i = 0; i < N; i++) {
        Phi.block(i*nx, 0, nx, nx) = matrix_power(A, i+1);
        //Phi.block<12, 12>(i * 12, 0) = matrix_power(A, i+1);
        for (int j = 0; j <= i; j++) {
            Gamma.block(i * 12, j * 8, 12, 8) = matrix_power(A, i-j) * B;
            //Gamma.block<12, 8>(i * 12, j * 8) = matrix_power(A, i-j) * B;
        }
    }*/
    
    
    real_t u_lb[N*nu];
    real_t u_ub[N*nu];
    for (int i(0); i<N; i++) 
    {
        for (int j(0); j<4; j++)
        {
            u_lb[8*i + 2*j+1] = umin[2*j+1];
            u_ub[8*i + 2*j+1] = umax[2*j+1];
        }
    }
    
    
    int nC = N*4;
    int nV = N*nu;
    
    Eigen::MatrixXd Ueq = Eigen::MatrixXd::Zero(nC, nV);                // Constraint on memory
    //Eigen::VectorXd beq = Eigen::VectorXd::Zero(nC);
    
    /*Ueq(0, 1) = 1;
    Ueq(1, 3) = 1;
    Ueq(2, 5) = 1;
    Ueq(3, 7) = 1;*/
        
    for (int i(0); i<N-1; i++)
    {
        Ueq.block(4*i, 8*i, 1, 16)   << 1-r[0], r[0], 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0;
        Ueq.block(4*i+1, 8*i, 1, 16) << 0, 0, 1-r[1], r[1], 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0;
        Ueq.block(4*i+2, 8*i, 1, 16) << 0, 0, 0, 0, 1-r[2], r[2], 0, 0, 0, 0, 0, 0, -1, 0, 0, 0;
        Ueq.block(4*i+3, 8*i, 1, 16) << 0, 0, 0, 0, 0, 0, 1-r[3], r[3], 0, 0, 0, 0, 0, 0, -1, 0;
    }
    
    real_t U_qp[nC * nV];
    real_t lbA_qp[nC], ubA_qp[nC];
    
    for (int i(0); i<nC; i++) 
    {
        lbA_qp[i] = 0;
        ubA_qp[i] = 0;
        for (int j(0); j<nV; j++) {U_qp[i*nV + j] = Ueq(i, j);}
    }
    
    //  H
    //Eigen::MatrixXd H = 2 * Gamma.transpose() * NQ * Gamma + NR;
    real_t H_qp[N * nu * N * nu];
    for (int i = 0; i<N*nu; i++) 
    {
        for (int j = 0; j < N * nu; j++) {
            H_qp[i*N*nu + j] = H(i, j);
        }
    }
    
    // QProblem setting
    QProblem qp(nV, nC);
    Options options;
    options.setToMPC();
    qp.setOptions(options);
    int nWSR = 20;
    
    // Initialization
    int step(0);
    real_t g_qp[N * nu];
    // PROCEDURE
    ros::Time T = ros::Time::now();
    while (step<N)
    {
        x0 << current_pose(0), current_pose(1), current_pose(2), current_euler(2), current_vel(0), current_vel(1), current_vel(2), current_yaw_vel, current_acc(0), current_acc(1), current_acc(2), current_yaw_acc;
        //Eigen::VectorXd g = 2 * Gamma.transpose() * NQ * (Phi * x0 - xref.head(12*N));
        Eigen::VectorXd g = 2 * Gamma.transpose() * NQ * (Phi * x0 - xref.segment(12*step, 12*(step+N)));
    
        for (int i = 0; i < N * nu; i++) {g_qp[i] = g(i);}

        //qp.init(H_qp, g_qp, U_qp, u_lb, u_ub, lbA_qp, ubA_qp, nWSR);
        qp.init(H_qp, g_qp, nullptr, u_lb, u_ub, nullptr, nullptr, nWSR);
        real_t u_opt[N * nu];
        qp.getPrimalSolution(u_opt);
        ROS_INFO_STREAM("Optimal u0 : " << u_opt[0] << " " << u_opt[1] << " " << u_opt[2] << " " << u_opt[3] << " " << u_opt[4] << " " << u_opt[5] << " " << u_opt[6] << " " << u_opt[7]);
    
        double yaw = current_euler(2);
        geometry_msgs::Twist input;
        input.linear.x = u_opt[1] * cos(yaw) + u_opt[3] * sin(yaw);                  //Invert Rotmatrix
        input.linear.y = -u_opt[1] * sin(yaw) + u_opt[3] * cos(yaw);
        input.linear.x = u_opt[5];
        input.angular.x = 0;
        input.angular.y = 0;
        input.angular.z = u_opt[7];
    
        ros::Time T = ros::Time::now();
        exe_pub.publish(input);
        do {}
        while (ros::Time::now() < T + ros::Duration((step+1)*delta_t));
        step++;
        
        // TIME REGULATOR
    }
}
