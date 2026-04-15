#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <map>
#include <string>

#include "quadrotor_mpc/linear_mpc.h"
#include <casadi/casadi.hpp>
#include <limits>


using namespace casadi;



linear_MPC::linear_MPC(bool gt, bool w)
{
    ground_truth = gt; 
    wind = w; 
    init_parameters();
}

linear_MPC::~linear_MPC() {}


Eigen::Vector3d linear_MPC::quaternion_to_euler(Eigen::Vector4d& q) const
{
    double roll = atan2(2*(q(0)*q(1) + q(2)*q(3)), 1 - 2*(q(1)*q(1) + q(2)*q(2)));
    double pitch = -M_PI/2 + 2*atan2(sqrt(1 + 2*(q(0)*q(2) - q(1)*q(3))), sqrt(1 - 2*(q(0)*q(2) - q(1)*q(3))));
    double yaw = atan2(2*(q(0)*q(3) - q(1)*q(2)), 1 - 2*(q(2)*q(2) + q(3)*q(3)));
    return Eigen::Vector3d(roll, pitch, yaw);
}


void linear_MPC::build_ABCD()
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


void linear_MPC::pose_callback(const geometry_msgs::PoseStamped::ConstPtr& pose)
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


void linear_MPC::euler_callback(const geometry_msgs::PoseStamped::ConstPtr& pose)
{
    current_ori(0) = pose->pose.orientation.w;
    current_ori(1) = pose->pose.orientation.x;
    current_ori(2) = pose->pose.orientation.y;
    current_ori(3) = pose->pose.orientation.z;
    double prev_yaw = current_yaw;
    current_euler = quaternion_to_euler(current_ori);
    current_yaw = current_euler(2);
    double prev_yaw_vel = current_yaw_vel;
    current_yaw_vel = (current_yaw - prev_yaw) / dt;
    current_yaw_acc = (current_yaw_vel - prev_yaw_vel) / dt;
}


/*void linear_MPC::euler_callback(const geometry_msgs::Vector3Stamped::ConstPtr& vec)
{
    double prev_yaw = current_euler(2);
    double prev_yaw_vel = current_yaw_vel;
    current_euler(0)  = vec->vector.x;
    current_euler(1) = vec->vector.y;
    current_euler(2)   = vec->vector.z;
    current_yaw = current_euler(2);
    current_yaw_vel = (current_yaw - prev_yaw) / dt;
    current_yaw_acc = (current_yaw_vel - prev_yaw_vel) / dt;
}*/


void linear_MPC::command_callback(const geometry_msgs::Twist::ConstPtr& twist)
{
    current_command(0) = twist->linear.x;
    current_command(1) = twist->linear.y;
    current_command(2) = twist->linear.z;
    current_command(3) = twist->angular.z;
}


void linear_MPC::traj_length_callback(const std_msgs::Int16::ConstPtr& msg)
{
    trajL = msg->data;
    xref = DM::zeros(nx, trajL);
    uref = DM::zeros(nu, trajL);
    ROS_INFO("traj length: %d", trajL);
}


void linear_MPC::traj_points_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        xref(0, i) = t->data[3*i];
        xref(3, i) = t->data[3*i+1];
        xref(6, i) = t->data[3*i+2];
    }
}


void linear_MPC::traj_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        xref(1, i) = t->data[3*i];
        xref(4, i) = t->data[3*i+1];
        xref(7, i) = t->data[3*i+2];
    }
}


void linear_MPC::traj_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        xref(2, i) = t->data[3*i];
        xref(5, i) = t->data[3*i+1];
        xref(8, i) = t->data[3*i+2];
    }
}


void linear_MPC::traj_yaws_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        xref(9, i) = t->data[i];
}


void linear_MPC::traj_yaws_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        xref(10, i) = t->data[i];
}


void linear_MPC::traj_yaws_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        xref(11, i) = t->data[i];
}


void linear_MPC::timestep_callback(const std_msgs::Float32::ConstPtr& dt)
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


void linear_MPC::ui_command_callback(const std_msgs::Int16::ConstPtr& msg)
{
    if (msg->data==13 || msg->data==17) 
    {
        if (msg->data == 17)
            build_fixed_point();
        build_solver();
        timer = std::make_shared<ros::Timer>(_nh.createTimer(ros::Duration(delta_t), &linear_MPC::run_trajectory, this));
        ROS_INFO("Timer created !");
    }
    else if (msg->data==14) 
    {
        ROS_INFO_STREAM("position: " << current_pose(0) << " " << current_pose(1) << " " << current_pose(2));
        ROS_INFO_STREAM("roll: " << current_euler(0) << ", pitch: " << current_euler(1) << ", yaw: " << current_euler(2));
    }
    else if (msg->data==15)
    {
        try
        {
            assert(trajL >= 3);
            std::cout << "length " << trajL << " REGISTERED TRAJECTORY:" << std::endl;
            for (int i(0); i<trajL; i++)
                std::cout << " "<< xref(0, i) <<" "<< xref(1, i) <<" "<< xref(2, i) <<" "<< xref(3, i) <<" "<< xref(4, i) <<" "<< xref(5, i) <<" "<< xref(6, i) <<" "<< xref(7, i) <<" "<< xref(8, i) <<" "<< xref(9, i) <<" "<< xref(10, i) << xref(11, i) << std::endl;
        }
        catch(...) {ROS_WARN("NO TRAJ RECORDED");}
    }
    else if (msg->data==16 || msg->data==18)
    {
        if (msg->data == 18)
            build_fixed_point(true);
        build_solver();
        iteration_test();
    }
    else if (msg->data==19)
    {
        ROS_INFO("STOPPING...");
        timer = nullptr;
        step = 0;
        input.linear.x = 0;
        input.linear.y = 0;
        input.linear.z = 0;
        input.angular.x = 0;
        input.angular.y = 0;
        input.angular.z = 0;
        exe_pub.publish(input);
        ROS_INFO("Timer destroyed !");
        
        result_traj_pub.publish(result_traj);
        result_traj.data.clear();
    }
}


void linear_MPC::init_parameters() 
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
    xmin = {-5, -3, -3, -5, -3, -3, 0.18, -3, -3, -std::numeric_limits<double>::max(), -M_PI, -2*M_PI};
    xmax = {5, 3, 3, 5, 3, 3, 5, 3, 3, std::numeric_limits<double>::max(), M_PI, 2*M_PI};
    _nh.getParam("umin", umin);
    _nh.getParam("umax", umax);
    
    _nh.getParam("method", method);
    ROS_INFO_STREAM("LOADED METHOD: " << method);
    
    _nh.getParam("P_factor", P_factor);
    
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
    
    
    current_pose << 0, 0, 0;
    current_euler << 0, 0, 0;
    current_ori << 0, 0, 0, 0;
    //xref = Eigen::VectorXd::Zero(nx*N);
    //uref = Eigen::VectorXd::Zero(nu*N);
    
    pose_subscriber    = _nh.subscribe<geometry_msgs::PoseStamped>("/ground_truth_to_tf/pose", 1, &linear_MPC::pose_callback, this);
    euler_subscriber   = _nh.subscribe<geometry_msgs::PoseStamped>("/ground_truth_to_tf/pose", 1, &linear_MPC::euler_callback, this);
    command_subscriber = _nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 1, &linear_MPC::command_callback, this);
    
    traj_length_subscriber = _nh.subscribe<std_msgs::Int16>("/trajectory_length", 1, &linear_MPC::traj_length_callback, this);
    traj_points_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_points", 1, &linear_MPC::traj_points_callback, this);
    traj_speeds_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_speeds", 1, &linear_MPC::traj_speeds_callback, this);
    traj_accs_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_accs", 1, &linear_MPC::traj_accs_callback, this);
    
    yaws_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_yaws", 1, &linear_MPC::traj_yaws_callback, this);
    yaws_speeds_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_yaws_speeds", 1, &linear_MPC::traj_yaws_speeds_callback, this);
    yaws_accs_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_yaws_accs", 1, &linear_MPC::traj_yaws_accs_callback, this);
    
    timestep_subscriber = _nh.subscribe<std_msgs::Float32>("/timestep", 1, &linear_MPC::timestep_callback, this);
    ui_command_subscriber = _nh.subscribe<std_msgs::Int16>("/command", 1, &linear_MPC::ui_command_callback, this);
    
    exe_pub = _nh.advertise<geometry_msgs::Twist>("/cmd_vel", 500, true);
    result_traj_pub = _nh.advertise<std_msgs::Float64MultiArray>("/result_poses", 1, true);
    
    build_ABCD();
    cost_matrices();
    N_cost_matrices();
}


void linear_MPC::cost_matrices()
{
    std::vector<double> Qvec;
    std::vector<double> Rvec;
    _nh.getParam("Q", Qvec);
    _nh.getParam("R", Rvec);
    Q = SX::zeros(nx, nx);
    R = SX::zeros(nu, nu);
    for (int i(0); i<nx; i++) {Q(i, i) = Qvec[i];}
    for (int i(0); i<nu; i++) {R(i, i) = Rvec[i];}
    P = P_factor * Q;
}


void linear_MPC::N_cost_matrices()
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


Eigen::MatrixXd linear_MPC::matrix_power(Eigen::MatrixXd M, int n)
{
    Eigen::MatrixXd S(M);
    for (int i(0); i<n-1; i++)
        S = M * S;
    return S;
}


void linear_MPC::constraints_matrices(){}



void linear_MPC::build_fixed_point(bool test, double speed, int margin)
{
    double coords[4];
    std::string symb[4] = {"x", "y", "z", "yaw"};
    for (int i(0); i<4; i++)
    {
        while (true)
        {
            try
            {
                std::cout << symb[i] << " position: "; 
                std::cin >> coords[i];
                break;
            }
            catch(...) {}
        }
    }
    Eigen::Vector3d pos;
    double yaw;
    if (test)
    {
        pos(0) = 0;
        pos(1) = 0;
        pos(2) = 0.2;
        yaw = 0;
    }
    else
    {
        pos = current_pose;
        yaw = current_yaw;
    }
    double distance = sqrt(pow(pos(0) - coords[0], 2) + pow(pos(1) - coords[1], 2) + pow(pos(2) - coords[2], 2));
    double yaw_distance = std::fmod(coords[3] - current_yaw + M_PI, 2*M_PI) - M_PI;
    int length = distance / (speed * delta_t);
    int full_length = length + margin;
    xref = DM::zeros(nx, full_length);
    uref = DM::zeros(nu, full_length);
    trajL = full_length;
    for (int i(0); i<length; i++)
    {
        xref(0, i) = pos(0) + i * (coords[0] - pos(0)) / length;
        xref(3, i) = pos(1) + i * (coords[1] - pos(1)) / length;
        xref(6, i) = pos(2) + i * (coords[2] - pos(2)) / length;
        xref(9, i) = yaw + i * yaw_distance / length;
        
        xref(1, i)  = (pos(0) - coords[0]) * speed / distance;
        xref(4, i)  = (pos(1) - coords[1]) * speed / distance;
        xref(7, i)  = (pos(2) - coords[2]) * speed / distance;
        xref(10, i) = yaw_distance * speed / distance;
        
        xref(2, i)  = 0;
        xref(5, i)  = 0;
        xref(8, i)  = 0;
        xref(11, i) = 0;
    }
    for (int i(length); i<trajL; i++)
    {
        xref(0, i) = coords[0];
        xref(3, i) = coords[1];
        xref(6, i) = coords[2];
        xref(9, i) = yaw + yaw_distance;
        
        xref(1, i)  = 0;
        xref(4, i)  = 0;
        xref(7, i)  = 0;
        xref(10, i) = 0;
        
        xref(2, i)  = 0;
        xref(5, i)  = 0;
        xref(8, i)  = 0;
        xref(11, i) = 0;
    }
}



void linear_MPC::build_solver()
{
    ROS_INFO("Building Solver...");

    double kp[4] = {lxy_kp, lxy_kp, lz_kp, az_kp};
    double ki[4] = {lxy_ki, lxy_ki, lz_ki, az_ki};
    double kd[4] = {lxy_kd, lxy_kd, lz_kd, az_kd};
    double r[4]  = {delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lz_tau), delta_t / (delta_t + az_tau)};
    
    SX X = SX::sym("X", nx, N+1);
    SX U = SX::sym("U", nu, N);
    SX X0 = SX::sym("X0", nx);
    SX U0 = SX::sym("U0", nu);
    //SX Xref = SX::sym("Xref", nx, N+1);
    SX Xref = SX::sym("Xref", nx*(N+1));
    
    SX cost = 0;
    std::vector<SX> constraints;
    constraints.push_back(X(Slice(), 0) - X0);
    
    for (int k = 0; k < N; k++) 
    {
        SX xk = X(Slice(), k);
        SX uk = U(Slice(), k);
        SX xkp1 = X(Slice(), k+1);
        SX err = xk - Xref(Slice(k*nx, (k+1)*nx));
        //err(Slice(), 9) = fmod(err(Slice(), 9), 2*M_PI);
        cost += mtimes(mtimes(err.T(), Q), err);
        cost += mtimes(mtimes(uk.T(), R), uk);
        //cost += SX::dot(xk - Xref(Slice(), k), SX::dot(Q, (xk - Xref(Slice(), k)))) + SX::dot(uk, SX::dot(R, uk));
        constraints.push_back(xkp1 - (mtimes(A, xk) + mtimes(B, uk)));
    }
    SX err = X(Slice(), N) - Xref(Slice((N-1)*nx, N*nx));
    cost += mtimes(mtimes(err.T(), P), err);
    
    // Recurrent constraints on inputs (1st order filter)
    constraints.push_back(U(Slice(), 0)(0) - r[0]*U0(1) - (1-r[0])*U0(0));        // U0 represents the current input
    constraints.push_back(U(Slice(), 0)(2) - r[1]*U0(3) - (1-r[1])*U0(2));
    constraints.push_back(U(Slice(), 0)(4) - r[2]*U0(5) - (1-r[2])*U0(4));
    constraints.push_back(U(Slice(), 0)(6) - r[3]*U0(7) - (1-r[3])*U0(6));
    for (int k = 0; k < N-1; k++)
    {
        constraints.push_back(U(Slice(), k+1)(0) - r[0]*U(Slice(), k)(1) - (1-r[0])*U(Slice(), k)(0));
        constraints.push_back(U(Slice(), k+1)(2) - r[1]*U(Slice(), k)(3) - (1-r[1])*U(Slice(), k)(2));
        constraints.push_back(U(Slice(), k+1)(4) - r[2]*U(Slice(), k)(5) - (1-r[2])*U(Slice(), k)(4));
        constraints.push_back(U(Slice(), k+1)(6) - r[3]*U(Slice(), k)(7) - (1-r[3])*U(Slice(), k)(6));
    }
    
    // Inequality Constraints on the N horizon
    std::vector<double> lbx;
    std::vector<double> ubx;
    for (int i(0); i<N+1; i++)
    {
        for (int j(0); j<nx; j++)
        {
            lbx.push_back(xmin[j]);
            ubx.push_back(xmax[j]);
        }
    }
    for (int i(0); i<N; i++)
    {
        for (int j(0); j<nu; j++)
        {
            lbx.push_back(umin[j]);
            ubx.push_back(umax[j]);
        }
    }
    
    // Building solver
    SX OPT_vars = vertcat(reshape(X, nx*(N+1), 1),
                          reshape(U, nu*N, 1));
    SX g = vertcat(constraints);
    //Function nlp = Function("nlp", {OPT_vars, X0, Xref}, {cost, g});
    //Function nlp = Function("nlp", {OPT_vars, X0, U0, Xref}, {cost, g});
    Dict opts;
    if (method == "ipopt")
    {
        opts["ipopt.print_level"] = 0;
        opts["ipopt.max_iter"] = 100;
    }
    //opts["hessian_approximation"] = "limited-memory";
    opts["print_time"] = 0;
    //opts["qpsol"] = "qpoases"; // or osqp
    solver = nlpsol("solver", method, {
        {"x", OPT_vars},
        {"f", cost},
        {"g", g},
        {"p", SX::vertcat({X0, U0, Xref})},            // Symbols relative to non optimized parameters
    }, opts);
    
    // args for CALL
    args["lbg"] = DM::zeros(nx*(N+1) + 4*N);
    args["ubg"] = DM::zeros(nx*(N+1) + 4*N);
    args["lbx"] = DM(lbx);
    args["ubx"] = DM(ubx);
    //args["x0"] = DM::zeros(nlp.n_in(0));  // init guess
    //args["p"] = DM::vertcat({x0, xref(Slice(0, nx))}); // X0, Xref
    //args["p"] = DM::vertcat({x0, u0, xref(Slice(), Slice(0, N))}); // X0, U0, Xref  <----- params
    
    std::vector<double> sp((N+1)*nx + N*nu, 0.0);
    sol_prev = sp;
    
    // Init guess
    x0 = DM::zeros(nx);
    u0 = DM::zeros(nu);
    
    u0(0) = current_command(0)*cos(current_yaw) - current_command(1)*sin(current_yaw);
    u0(1) = u0(0);
    u0(2) = current_command(0)*sin(current_yaw) + current_command(1)*cos(current_yaw);
    u0(3) = u0(2);
    u0(4) = current_command(2);
    u0(5) = u0(4);
    u0(6) = current_command(3);
    u0(7) = u0(6);
    
    xref_vec = reshape(xref, trajL*nx, 1);
    xref_vec_final = DM::zeros(nx*(N+1), 1);
    for (int i(0); i<N+1; i++)
    {
        xref_vec_final(i*nx)   = xref_vec(nx*trajL-nx);
        xref_vec_final(i*nx+3) = xref_vec(nx*trajL-nx+3);
        xref_vec_final(i*nx+6) = xref_vec(nx*trajL-nx+6);
        xref_vec_final(i*nx+9) = xref_vec(nx*trajL-nx+9);
    }
    
    std_msgs::MultiArrayDimension dim1, dim2;
    result_traj.layout.dim.push_back(dim1);
    result_traj.layout.dim.push_back(dim2);
    result_traj.layout.dim[0].label = "height";
    result_traj.layout.dim[1].label = "width";
    result_traj.layout.dim[0].size = trajL;
    result_traj.layout.dim[1].size = 12;
    result_traj.layout.dim[0].stride = trajL*12;
    result_traj.layout.dim[1].stride = 12;
    
    ROS_INFO("SOLVER BUILT !");
}


void linear_MPC::iteration_test()
{
    ROS_INFO("Iteration test...");

    // Arbitrary init conditions
    x0 = DM::zeros(nx);
    u0 = DM::zeros(nu);
    x0(6) = 0.2;
    ROS_INFO_STREAM("traj length: " << trajL);
    DM xref_vec = reshape(xref, nx*trajL, 1);
    std::vector<double> x0_guess((N+1)*nx + N*nu, 0.0);
    
    int test_step = 0;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    
    do
    {
        /*std::cout << "init state: " << x0(0) << " " << x0(1) << " " << x0(2) << " " << x0(3) << " " << x0(4) << " " << x0(5) << " " << x0(6) << " " << x0(7) << " " << x0(8) << " " << x0(9) << " " << x0(10) << " " << x0(11) << std::endl;
        std::cout << "init control: " << u0(0) << " " << u0(1) << " " << u0(2) << " " << u0(3) << " " << u0(4) << " " << u0(5) << " " << u0(6) << " " << u0(7) << std::endl;*/
    
        args["x0"] = DM(x0_guess);
        //args["p"] = DM::vertcat({x0, xref(Slice(), Slice(step, step+N))}); // X0, Xref
        //DM xref_part = xref(Slice(), Slice(0, N)).reshape(N*nx, 1);
        try {args["p"] = DM::vertcat({x0, u0, xref_vec(Slice(nx*test_step, (N+1+test_step)*nx))});} // X0, U0, Xref(step)    <----- params
        catch(...) {break;}
    
        std::map<std::string, DM> res = solver(args); // <------------------------------------------------------------------------
        DM sol = res.at("x");
        //std::cout << "SOL:\n" << sol.rows() << std::endl;
        DM test_x_opt = sol(Slice(0, (N+1)*nx));
        DM test_u_opt = sol(Slice((N+1)*nx, (N+1)*nx+N*nu));
    
        for (int i(0); i<N+1; i++)
            ROS_INFO_STREAM("x_opt_pred " << i+1 << ": " << std::endl << test_x_opt(nx*i) <<" "<< test_x_opt(nx*i+1) <<" "<< test_x_opt(nx*i+2) <<" "<< test_x_opt(nx*i+3) <<" "<< test_x_opt(nx*i+4) <<" "<< test_x_opt(nx*i+5) <<" "<< test_x_opt(nx*i+6) <<" "<< test_x_opt(nx*i+7) <<" "<< test_x_opt(nx*i+8) <<" "<< test_x_opt(nx*i+9) <<" "<< test_x_opt(nx*i+10) <<" "<< test_x_opt(nx*i+11));
        for (int i(0); i<N; i++)
            ROS_INFO_STREAM("u_opt_pred " << i+1 << ": " << std::endl << test_u_opt(nu*i) <<" "<< test_u_opt(nu*i+1) <<" "<< test_u_opt(nu*i+2) <<" "<< test_u_opt(nu*i+3) <<" "<< test_u_opt(nu*i+4) <<" "<< test_u_opt(nu*i+5) <<" "<< test_u_opt(nu*i+6) <<" "<< test_u_opt(nu*i+7));
        
        x0 = test_x_opt(Slice(nx, 2*nx));
        u0 = test_u_opt(Slice(0, nu));
        
        // Warm guess
        sol_prev = std::vector<double>(sol->begin(), sol->end());
        for (int i = 0; i < N; i++) 
        {
            for (int j(0); j<nx; j++)
                x0_guess[i*nx + j] = (i < N) ? sol_prev[(i+1)*nx + j] : sol_prev[N*nx + j];
            for (int j(0); j<nu; j++)
                x0_guess[(N+1)*nx + i*nu + j] = (i < N-1) ? sol_prev[(N+1)*nx + (i+1)*nu + j] : sol_prev[(N+1)*nx + (N-1)*nu];
        }
        
        //std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000*delta_t)));
        test_step++;
        
    } while (true);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    ROS_INFO_STREAM("Run time = " << elapsed_seconds.count() << "s\n");
    ROS_INFO("Test ends !");
}



void linear_MPC::run_trajectory(const ros::TimerEvent&)
{   
    //auto now = std::chrono::steady_clock::now();
    
    x0(0) = current_pose(0);
    x0(1) = current_vel(0);
    x0(2) = current_acc(0);
    x0(3) = current_pose(1);
    x0(4) = current_vel(1);
    x0(5) = current_acc(1);
    x0(6) = current_pose(2);
    x0(7) = current_vel(2);
    x0(8) = current_acc(2);
    x0(9) = current_yaw;
    x0(10) = current_yaw_vel;
    x0(11) = current_yaw_acc;
    
    for (int i(0); i<3; i++) {result_traj.data.push_back(static_cast<double>(x0(3*i)));}
    result_traj.data.push_back(current_euler(0));
    result_traj.data.push_back(current_euler(1));
    result_traj.data.push_back(static_cast<double>(x0(9)));

    // Refining CALL args (Warm init)
    std::vector<double> x0_guess((N+1)*nx + N*nu, 0.0);
    for (int i = 0; i < N; i++) 
    {
        for (int j(0); j<nx; j++)
            x0_guess[i*nx + j] = (i < N) ? sol_prev[(i+1)*nx + j] : sol_prev[N*nx + j];
        for (int j(0); j<nu; j++)
            x0_guess[(N+1)*nx + i*nu + j] = (i < N-1) ? sol_prev[(N+1)*nx + (i+1)*nu + j] : sol_prev[(N+1)*nx + (N-1)*nu];
    }
    args["x0"] = DM(x0_guess);
    try {args["p"] = DM::vertcat({x0, u0, xref_vec(Slice(nx*step, (N+1+step)*nx))});} // X0, U0, Xref(step)    <----- params
    catch(...) {args["p"] = DM::vertcat({x0, u0, xref_vec_final});}
        
    // CALL
    std::map<std::string, DM> res = solver(args); // <------------------------------------------------------------------------
    DM sol = res.at("x");
    u_opt = sol(Slice((N+1)*nx, (N+1)*nx+nu));
        
    // PUBLISHING IN STABLE FRAME
    double lx = static_cast<double>(u_opt(1));
    double ly = static_cast<double>(u_opt(3));
    double lz = static_cast<double>(u_opt(5));
    double az = static_cast<double>(u_opt(7));
    input.linear.x = lx * cos(current_yaw) + ly * sin(current_yaw);                  //Invert Rotmatrix
    input.linear.y = -lx * sin(current_yaw) + ly * cos(current_yaw);
    input.linear.z = lz;
    input.angular.x = 0;
    input.angular.y = 0;
    input.angular.z = az;
    exe_pub.publish(input);  // <--------------------------------------------------------------------------------
        
    // Actual last input
    u0 = u_opt;
    sol_prev = std::vector<double>(sol->begin(), sol->end());
    step++;
    
    /*std::cout << std::endl;    
    ROS_INFO_STREAM("Prev u0 memory : " << u_opt(0) << " " << u_opt(2) << " " << u_opt(4) << " " << u_opt(6) << " time: " << ros::Time::now());
    ROS_INFO_STREAM("Optimal sent u0 : " << u_opt(1) << " " << u_opt(3) << " " << u_opt(5) << " " << u_opt(7));*/
}
