#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <map>
#include <string>

#include "quadrotor_mpc/full_mpc.h"


using namespace casadi;



MPC::MPC(bool gt, bool w)
{
    ground_truth = gt; 
    wind = w; 
    init_parameters();
}

MPC::~MPC() {}


Eigen::Vector3d MPC::quaternion_to_euler(Eigen::Vector4d& q) const
{
    double roll = atan2(2*(q(0)*q(1) + q(2)*q(3)), 1 - 2*(q(1)*q(1) + q(2)*q(2)));
    double pitch = -M_PI/2 + 2*atan2(sqrt(1 + 2*(q(0)*q(2) - q(1)*q(3))), sqrt(1 - 2*(q(0)*q(2) - q(1)*q(3))));
    double yaw = atan2(2*(q(0)*q(3) - q(1)*q(2)), 1 - 2*(q(2)*q(2) + q(3)*q(3)));
    return Eigen::Vector3d(roll, pitch, yaw);
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
    current_euler = quaternion_to_euler(current_ori);
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


void MPC::euler_callback(const geometry_msgs::PoseStamped::ConstPtr& pose)
{
    current_ori(0) = pose->pose.orientation.w;
    current_ori(1) = pose->pose.orientation.x;
    current_ori(2) = pose->pose.orientation.y;
    current_ori(3) = pose->pose.orientation.z;
    double prev_x = current_euler(0);
    double prev_y = current_euler(1);
    double prev_z = current_euler(2);
    current_euler = quaternion_to_euler(current_ori);
    double prev_vel_x = current_euler_vel(0);
    double prev_vel_y = current_euler_vel(1);
    double prev_vel_z = current_euler_vel(2);
    current_euler_vel(0) = (current_euler(0) - prev_x) / dt;
    current_euler_vel(1) = (current_euler(1) - prev_y) / dt;
    current_euler_vel(2) = (current_euler(2) - prev_z) / dt;
    current_euler_acc(0) = (current_euler_vel(0) - prev_vel_x) / dt;
    current_euler_acc(1) = (current_euler_vel(1) - prev_vel_y) / dt;
    current_euler_acc(2) = (current_euler_vel(2) - prev_vel_z) / dt;
}


/*void MPC::euler_callback(const geometry_msgs::Vector3Stamped::ConstPtr& vec)
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
    xref = DM::zeros(nx, trajL);
    uref = DM::zeros(nu, trajL);
    ROS_INFO("traj length: %d", trajL);
}


void MPC::traj_points_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        xref(0, i) = t->data[3*i];
        xref(3, i) = t->data[3*i+1];
        xref(6, i) = t->data[3*i+2];
    }
}


void MPC::traj_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        xref(1, i) = t->data[3*i];
        xref(4, i) = t->data[3*i+1];
        xref(7, i) = t->data[3*i+2];
    }
}


void MPC::traj_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        xref(2, i) = t->data[3*i];
        xref(5, i) = t->data[3*i+1];
        xref(8, i) = t->data[3*i+2];
    }
}


void MPC::traj_yaws_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        xref(9, i) = t->data[i];
}


void MPC::traj_yaws_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        xref(10, i) = t->data[i];
}


void MPC::traj_yaws_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        xref(11, i) = t->data[i];
}


void MPC::timestep_callback(const std_msgs::Float32::ConstPtr& dt)
{
    delta_t = dt->data;
    ROS_INFO("time step: %f", delta_t);
}


void MPC::ui_command_callback(const std_msgs::Int16::ConstPtr& msg)
{
    if (msg->data==13 || msg->data==17) 
    {
        if (msg->data == 17)
            build_fixed_point();
        build_solver();
        timer = std::make_shared<ros::Timer>(_nh.createTimer(ros::Duration(delta_t), &MPC::run_trajectory, this));
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
            {
                for (int j(0); j<nx; j++) {std::cout <<" "<< xref(j, i) <<" ";}
                std::cout << std::endl;
            }
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
    else if (msg->data==21) 
    {
        debug_mode = !debug_mode;
        if (debug_mode)
            debug();
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
    
    _nh.getParam("controller/twist/angular/xy/k_p", axy_kp);
    _nh.getParam("controller/twist/angular/xy/k_i", axy_ki);
    _nh.getParam("controller/twist/angular/xy/k_d", axy_kd);
    _nh.getParam("controller/twist/angular/xy/time_constant", axy_tau);
    
    _nh.getParam("controller/twist/angular/z/k_p", az_kp);
    _nh.getParam("controller/twist/angular/z/k_i", az_ki);
    _nh.getParam("controller/twist/angular/z/k_d", az_kd);
    _nh.getParam("controller/twist/angular/z/time_constant", az_tau);
    
    _nh.getParam("delta_t", delta_t);
    
    gamma_x = lxy_kp + lxy_ki*delta_t;                               //+ lxy_kd/delta_t;
    gamma_y = lxy_kp + lxy_ki*delta_t;                               //+ lxy_kd/delta_t;
    gamma_z = lz_kp + lz_ki*delta_t;                                 //+ lz_kd/delta_t;
    gamma_wx = axy_kp + axy_ki*delta_t;                            //+ axy_kd/delta_t;
    gamma_wy = axy_kp + axy_ki*delta_t;                           //+ axy_kd/delta_t;
    gamma_wz = az_kp + az_ki*delta_t;                               //+ az_kd/delta_t;
    
    r[0] = delta_t / (delta_t + lxy_tau);
    r[1] = delta_t / (delta_t + lxy_tau);
    r[2] = delta_t / (delta_t + lz_tau);
    r[3] = delta_t / (delta_t + axy_tau);
    r[4] = delta_t / (delta_t + axy_tau);
    r[5] = delta_t / (delta_t + az_tau);
    
    std::vector<double> x_test;
    _nh.getParam("x_test", x_test);
    x0 = DM(x_test);
    x0(nx-2) = std::numeric_limits<double>::quiet_NaN();
    x0(nx-1) = std::numeric_limits<double>::quiet_NaN();
    
    std::vector<double> Ivec, OIvec;
    std::vector<double> CoGvec;
    I = DM::zeros(3, 3);
    OI = DM::zeros(3, 3);
    CoG = DM::zeros(3);
    
    _nh.getParam("mass", mass);
    _nh.getParam("IM", Ivec);
    _nh.getParam("OIM", OIvec);
    _nh.getParam("CoG", CoGvec);
    I = reshape(DM(Ivec), 3, 3);
    OI = reshape(DM(OIvec), 3, 3);
    
    _nh.getParam("Horizon", N);
    xmin = {-5, -3, -3, -5, -3, -3, 0.18, -3, -3, -M_PI/2, -2*M_PI, -4*M_PI, -M_PI/2, -2*M_PI, -4*M_PI, -std::numeric_limits<double>::max(), -M_PI, -2*M_PI, -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
    xmax = {5, 3, 3, 5, 3, 3, 5, 3, 3, M_PI/2, 2*M_PI, 4*M_PI, M_PI/2, 2*M_PI, 4*M_PI, std::numeric_limits<double>::max(), M_PI, 2*M_PI, std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    _nh.getParam("umin_nl", umin);
    _nh.getParam("umax_nl", umax);
    
    _nh.getParam("method", method);
    ROS_INFO_STREAM("LOADED METHOD: " << method);
    
    _nh.getParam("P_factor", P_factor);
    std::cout << "P factor: " << P_factor << std::endl;
    
    /*_nh.getParam("K", Kvec);
    K = DM::zeros(nu, nx);
    for (int i(0); i<nu; i++) 
    {
        for (int j(0); j<nx; j++)
        {
            K(i, j) = Kvec[i*nx + j];
            std::cout << K(i, j) << " ";
        }
        std::cout << std::endl;
    }*/
    
    
    current_pose << 0, 0, 0.2;
    current_euler << 0, 0, 0;
    current_ori << 0, 0, 0, 0;
    //xref = Eigen::VectorXd::Zero(nx*N);
    //uref = Eigen::VectorXd::Zero(nu*N);
    
    pose_subscriber    = _nh.subscribe<geometry_msgs::PoseStamped>("/ground_truth_to_tf/pose", 1, &MPC::pose_callback, this);
    euler_subscriber   = _nh.subscribe<geometry_msgs::PoseStamped>("/ground_truth_to_tf/pose", 1, &MPC::euler_callback, this);
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
    result_traj_pub = _nh.advertise<std_msgs::Float64MultiArray>("/result_poses", 1, true);
    
    cost_matrices();
    //N_cost_matrices();
}


void MPC::cost_matrices()
{
    std::vector<double> Qvec;
    std::vector<double> Rvec;
    _nh.getParam("Qnl", Qvec);
    _nh.getParam("Rnl", Rvec);
    Q = SX::zeros(nx, nx);
    R = SX::zeros(nu, nu);
    for (int i(0); i<nx; i++) {Q(i, i) = Qvec[i];}
    for (int i(0); i<nu; i++) {R(i, i) = Rvec[i];}
    P = P_factor * Q;
}


Eigen::MatrixXd MPC::matrix_power(Eigen::MatrixXd M, int n)
{
    Eigen::MatrixXd S(M);
    for (int i(0); i<n-1; i++)
        S = M * S;
    return S;
}


void MPC::constraints_matrices(){}



void MPC::build_fixed_point(bool test, double speed, int margin)
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
        yaw = current_euler(2);
    }
    double distance = sqrt(pow(pos(0) - coords[0], 2) + pow(pos(1) - coords[1], 2) + pow(pos(2) - coords[2], 2));
    double yaw_distance = std::fmod(coords[3] - current_euler(2) + M_PI, 2*M_PI) - M_PI;
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





// CURRENT STATE AND COMMAND SETTING

void MPC::init_x0_u0(bool init_command)
{
    // Current state
    x0 = DM::zeros(nx);
    x0(0) = current_pose(0);
    x0(1) = current_vel(0);
    x0(2) = current_acc(0);
    x0(3) = current_pose(1);
    x0(4) = current_vel(1);
    x0(5) = current_acc(1);
    x0(6) = current_pose(2);
    x0(7) = current_vel(2);
    x0(8) = current_acc(2);
    x0(9) = current_euler(0);
    x0(10) = current_euler_vel(0);
    x0(11) = current_euler_acc(0);
    x0(12) = current_euler(1);
    x0(13) = current_euler_vel(1);
    x0(14) = current_euler_acc(1);
    x0(15) = current_euler(2);
    x0(16) = current_euler_vel(2);
    x0(17) = current_euler_acc(2);
    
    current_body_twist_w = forward_body_twist_w(current_euler(0), current_euler(1), current_euler(2), current_euler_vel(0), current_euler_vel(1), current_euler_vel(2));
    x0(18) = current_body_twist_w(0);
    x0(19) = current_body_twist_w(1);
    
    if (init_command)
    {
        // Steady state command guess
        u0 = DM::zeros(nu);
        u0(0) = current_command(0)*cos(current_euler(2)) - current_command(1)*sin(current_euler(2));
        u0(1) = u0(0);
        u0(2) = current_command(0)*sin(current_euler(2)) + current_command(1)*cos(current_euler(2));
        u0(3) = u0(2);
        u0(4) = current_command(2);
        u0(5) = u0(4);
        u0(6) = current_command(3);
        u0(7) = u0(6);
        Eigen::Vector3d xy_command = forward_body_twist_command_w(x0, u0);   // Here u0(8) and u0(9) are zeros (useless).
        u0(8) = xy_command(0);
        u0(9) = xy_command(1);
    }
}



// Non Linear Functions


Eigen::Vector3d MPC::forward_body_twist_w(double& roll, double& pitch, double& yaw, double& rolld, double& pitchd, double& yawd)
{
    Eigen::Matrix3d m1, m2;
    
    // Getting current twist from current euler angles and their rates
    m1 << 1,          0,          -sin(pitch),
          0,  cos(roll), cos(pitch)*sin(roll),
          0, -sin(roll), cos(pitch)*cos(roll);
    Eigen::Vector3d eulerd;
    eulerd << rolld, pitchd, yawd;
    Eigen::Vector3d twist_w = m1 * eulerd;
    
    // Getting body version
    m2 << cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll),
          sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll),
                   -sin(pitch),                              sin(roll)*cos(pitch),                               cos(roll)*cos(pitch);
    return m2.transpose() * twist_w;
}


Eigen::Vector3d MPC::forward_body_twist_command_w(DM& state, DM& command)
{
    // Only for setting init command in a steady state configuration

    double cart_speeds[3] = {static_cast<double>(state(1)), static_cast<double>(state(4)), static_cast<double>(state(7))};
    std::vector<double> commands = static_cast<std::vector<double>>(command);
    double acceleration_command_x = -gamma_x * cart_speeds[0] + (1-r[0])*gamma_x*commands[0] + r[0]*gamma_x*commands[1];
    double acceleration_command_y = -gamma_y * cart_speeds[1] + (1-r[1])*gamma_y*commands[2] + r[1]*gamma_y*commands[3];
    double acceleration_command_z = -gamma_z * cart_speeds[2] + (1-r[2])*gamma_z*commands[4] + r[2]*gamma_z*commands[5] + g;
    Eigen::Vector3d vec;
    vec << acceleration_command_x, acceleration_command_y, acceleration_command_z;
    
    Eigen::Matrix3d m;
    double roll(static_cast<double>(state(9))), pitch(static_cast<double>(state(12))), yaw(static_cast<double>(state(15)));
    m << cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll),
         sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll),
                  -sin(pitch),                              sin(roll)*cos(pitch),                               cos(roll)*cos(pitch);
    Eigen::Vector3d vec_body = m.transpose() * vec;
    
    Eigen::Vector3d xy_command;
    xy_command(0) = -vec_body(1) / g;
    xy_command(1) =  vec_body(0) / g;
    return xy_command;
}





/*
// NONLINEAR SYSTEM FUNCTIONS (CasADi semantics !)


SX MPC::VecToso3(SX vec) const
{
    SX so3 = SX::zeros(3, 3);
    so3(0, 1) = -vec(2);
    so3(0, 2) =  vec(1);
    so3(1, 0) =  vec(2);
    so3(1, 2) = -vec(0);
    so3(2, 0) = -vec(1);
    so3(2, 1) =  vec(0);
    return so3;
}


SX MPC::euler_to_quaternion(SX& roll, SX& pitch, SX& yaw) const
{
    SX q = SX::zeros(4);
    q(1) = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2);
    q(2) = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2);
    q(3) = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2);
    q(0) = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2);
    return q;
}


SX MPC::euler_to_rotmatrix(SX& roll, SX& pitch, SX& yaw) const
{
    SX m = SX::zeros(3, 3);
    m(0, 0) = cos(yaw)*cos(pitch);
    m(0, 1) = cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll);
    m(0, 2) = cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll);
    m(1, 0) = sin(yaw)*cos(pitch);
    m(1, 1) = sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll);
    m(1, 2) = sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll);
    m(2, 0) = -sin(roll);
    m(2, 1) = sin(roll)*cos(pitch);
    m(2, 2) = cos(roll)*cos(pitch);
    return m;
}


SX MPC::eulerd2w(SX& euler, SX& eulerd) const
{
    SX roll(euler(0)), pitch(euler(1)), yaw(euler(2));
    SX m = SX::zeros(3, 3);
    m(0, 0) = 1;
    m(0, 1) = 0;
    m(0, 2) = -sin(pitch);
    m(1, 0) = 0;
    m(1, 1) = cos(roll);
    m(1, 2) = cos(pitch)*sin(roll);
    m(2, 0) = 0;
    m(2, 1) = -sin(roll);
    m(2, 2) = cos(pitch)*cos(roll);
    return mtimes(m, eulerd);
}


SX MPC::wd2eulerdd(SX& euler, SX& eulerd, SX& wd) const
{
    SX roll(euler(0)), pitch(euler(1)), yaw(euler(2));
    SX rolld(eulerd(0)), pitchd(eulerd(1)), yawd(eulerd(2));
    std::vector<SX> m_vec = {1, 0, -sin(pitch), 0, cos(roll), cos(pitch)*sin(roll), 0, -sin(roll), cos(pitch)*cos(roll)};
    std::vector<SX> md_vec = {0, 0, -pitchd*cos(pitch), 0, -rolld*sin(roll), rolld*cos(pitch)*cos(roll) - pitchd*sin(roll)*sin(pitch), 0, -rolld*cos(roll), -rolld*cos(pitch)*sin(roll) - pitchd*cos(roll)*sin(pitch)};
    SX m = SX::zeros(3, 3);
    SX md = SX::zeros(3, 3);
    m = SX::reshape(DM(m_vec), 3, 3);
    md = SX::reshape(DM(md_vec), 3, 3);
    //SX m = SX({{1, 0, -sin(pitch)}, {0, cos(roll), cos(pitch)*sin(roll)}, {0, -sin(roll), cos(pitch)*cos(roll)}});
    //SX md = SX({{0, 0, -pitchd*cos(pitch)}, {0, -rolld*sin(roll), rolld*cos(pitch)*cos(roll) - pitchd*sin(roll)*sin(pitch)}, {0, -rolld*cos(roll), -rolld*cos(pitch)*sin(roll) - pitchd*cos(roll)*sin(pitch)}});
    return mtimes(inv(m), wd - mtimes(md, eulerd));
}


SX MPC::load_factor(SX& euler_angles) const
{
    SX roll(euler_angles(0)), pitch(euler_angles(1)), yaw(euler_angles(2));
    SX q = euler_to_quaternion(roll, pitch, yaw);
    return 1 / (q(0)*q(0) - q(1)*q(1) - q(2)*q(2) + q(3)*q(3));
}


SX MPC::twist_body_angular(SX& state) const
{
    SX roll(state(9)), pitch(state(12)), yaw(state(15));
    SX rolld(state(10)), pitchd(state(13)), yawd(state(16));
    SX euler(horzcat(roll, horzcat(pitch, yaw))); 
    SX eulerd(vertcat(rolld, vertcat(pitchd, yawd)));
    SX w = eulerd2w(euler, eulerd);
    SX m = euler_to_rotmatrix(roll, pitch, yaw);
    return mtimes(m.T(), w);
}


SX MPC::toBody(SX& state, SX& vec) const
{
  SX m = SX::zeros(3, 3);
  SX roll(state(9)), pitch(state(12)), yaw(state(15));
  m(0, Slice(0, 3)) = horzcat(cos(yaw)*cos(pitch), horzcat(cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll)));
  m(1, Slice(0, 3)) = horzcat(sin(yaw)*cos(pitch), horzcat(sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll)));
  m(2, Slice(0, 3)) = horzcat(-sin(roll), horzcat(sin(roll)*cos(pitch), cos(roll)*cos(pitch)));
  SX vec_body_x = m(0, 0) * vec(0) + m(1, 0) * vec(1) + m(2, 0) * vec(2);
  SX vec_body_y = m(0, 1) * vec(0) + m(1, 1) * vec(1) + m(2, 1) * vec(2);
  SX vec_body_z = m(0, 2) * vec(0) + m(1, 2) * vec(1) + m(2, 2) * vec(2);
  SX vec_body = horzcat(vec_body_x, horzcat(vec_body_y, vec_body_z));
  return vec_body;
}


SX MPC::get_acceleration_commands(SX& state, SX& command) const
{
    // Stating passive commands and abstract states are non zero.
    SX cart_speeds[3] = {state(1), state(4), state(7)};
    SX acceleration_command_x = -gamma_x * cart_speeds[0] + (1-r[0])*gamma_x*command(0) + r[0]*gamma_x*command(1);              //            /!\ Flight must be initialized !
    SX acceleration_command_y = -gamma_y * cart_speeds[1] + (1-r[1])*gamma_y*command(2) + r[1]*gamma_y*command(3);
    SX acceleration_command_z = -gamma_z * cart_speeds[2] + (1-r[2])*gamma_z*command(4) + r[2]*gamma_z*command(5) + g;
    return horzcat(acceleration_command_x, horzcat(acceleration_command_y, acceleration_command_z));
}


SX MPC::get_torques(SX& acceleration_command, SX& state, SX& command, SX& twist_body) const
{
    SX acceleration_command_body = toBody(state, acceleration_command);
    SX acceleration_command_body_x = acceleration_command_body(0);
    SX acceleration_command_body_y = acceleration_command_body(1);
    SX uwz_pass(command(6)), uwz(command(7));
    SX euler  = horzcat(state(9), horzcat(state(12), state(15)));
    SX eulerd = vertcat(state(10), vertcat(state(13), state(16)));
    SX wz = eulerd2w(euler, eulerd)(2);
    double I1(I(0, 0)), I2(I(1, 1)), I3(I(2, 2));
    SX uwx_body(-acceleration_command_body_y/g), uwy_body(acceleration_command_body_x/g);      // Induced commands
    SX uwx_body_pass(command(8)), uwy_body_pass(command(9));                                   // Inner passive commands
    SX tbx(twist_body(0)), tby(twist_body(1));
    SX prev_tbx(state(nx-2)), prev_tby(state(nx-1));
    SX torque_x = I1 * gamma_wx * ((1-r[3]) * uwx_body_pass + r[3] * uwx_body + axy_kd * ((uwx_body - uwx_body_pass)/dt - (tbx - prev_tbx)));  //  /!\ Flight must be initialized !
    SX torque_y = I2 * gamma_wy * ((1-r[4]) * uwy_body_pass + r[4] * uwy_body + axy_kd * ((uwy_body - uwy_body_pass)/dt - (tby - prev_tby)));
    SX torque_z = I3 * gamma_wz * (-wz + (1-r[5]) * uwz_pass + r[5] * uwz);
    return horzcat(torque_x, horzcat(torque_y, torque_z));
}


SX MPC::get_force(SX& state, SX acceleration_command_z) const
{
    SX euler_angles = horzcat(state(9), horzcat(state(12), state(15)));
    return mass * ((acceleration_command_z - g) * load_factor(euler_angles) + g);
}


SX MPC::AdjointTwist(SX& twist) const
{
    SX v(horzcat(twist(0), horzcat(twist(1), twist(2)))), w(horzcat(twist(3), horzcat(twist(4), twist(5))));
    SX AT = SX::zeros(6, 6);
    AT(Slice(0, 3), Slice(0, 3)) = VecToso3(w);
    AT(Slice(0, 3), Slice(3, 6)) = SX::zeros(3, 3);
    AT(Slice(3, 6), Slice(0, 3)) = VecToso3(v);
    AT(Slice(3, 6), Slice(3, 6)) = VecToso3(w);
    return AT;
}


SX MPC::get_euler_accs(SX& state, SX& torques, SX& rel_force) const
{
    SX speeds = SX::zeros(6);
    for (int i(0); i<6; i++) {speeds(i) = state(3*i+1);}
    SX wrench = SX::zeros(6);
    for (int i(0); i<3; i++) {wrench(i) = torques(i);}
    SX G = SX::zeros(6, 6);
    G(Slice(0, 3), Slice(0, 3)) = OI;
    G(Slice(0, 3), Slice(3, 6)) = mass * VecToso3(CoG);
    G(Slice(3, 6), Slice(0, 3)) = -mass * VecToso3(CoG);
    G(Slice(3, 6), Slice(3, 6)) = mass * SX::eye(3);
    SX all_accs = SX::zeros(6);
    all_accs = inv(G) * (wrench + transpose(AdjointTwist(speeds)) * G * speeds);
    SX force_offset = SX::zeros(3);
    double CoG0(CoG(0)), CoG1(CoG(1));
    force_offset(0) = CoG1*rel_force;
    force_offset(1) = -CoG0*rel_force;
    SX all_wd = all_accs(Slice(0, 3)) - force_offset;
    SX euler = horzcat(state(9), horzcat(state(12), state(15)));
    SX eulerd = vertcat(state(10), vertcat(state(13), state(16)));
    SX all_euler_accs = wd2eulerdd(euler, eulerd, all_wd);
    return all_euler_accs;
}


SX MPC::get_cartesian_accs(SX& state, SX& rel_force_command) const
{
    SX roll(state(9)), pitch(state(12)), yaw(state(15));
    SX acc_x = rel_force_command/mass * (cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll));
    SX acc_y = rel_force_command/mass * (sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll));
    SX acc_z = rel_force_command/mass * cos(pitch)*cos(roll) - g;
    return vertcat(acc_x, vertcat(acc_y, acc_z));
}


SX MPC::state_vec_from_acc(SX& state, SX& acc_vec, SX& twist_body) const
{
    SX state_vec = SX::zeros(nx);
    for (int i(0); i<6; i++)
    {
        state_vec(3*i)   = state(3*i) + delta_t * state(3*i+1) + delta_t*delta_t/2 * state(3*i+2);
        state_vec(3*i+1) = state(3*i+1) + delta_t * state(3*i+2);
        state_vec(3*i+2) = acc_vec(i);
    }
    state_vec(nx-2) = twist_body(0);       // Setting previous twist body
    state_vec(nx-1) = twist_body(1);
    return state_vec;
}


SX MPC::forward(SX& x, SX& u) const
{
    SX euler = vertcat(x(9), vertcat(x(12), x(15)));
    SX eulerd = vertcat(x(10), vertcat(x(13), x(16)));
    SX twist = eulerd2w(euler, eulerd);
    SX tba = toBody(x, twist);

    SX acc_commands = get_acceleration_commands(x, u);
    SX planar_torques = get_torques(acc_commands, x, u, tba);
    SX relative_z_force = get_force(x, acc_commands(2));
    SX output_cart_acc = get_cartesian_accs(x, relative_z_force);
    SX output_euler_acc = get_euler_accs(x, planar_torques, relative_z_force);
    SX output_acc = vertcat(output_cart_acc, output_euler_acc);
    SX output = state_vec_from_acc(x, output_acc, tba);
    return output;
}

// END OF NONLINEAR SYSTEM FUNCTIONS*/


void MPC::debug(bool initialization)
{
    DM state = x0;
    std::vector<double> c;
    int i(0);
    double ui;
    if (!initialization)
    {
        do {
            std::cout << "Enter passive command vector " << i << ": ";
            try {
                std::cin >> ui;
                c.push_back(ui);
                c.push_back(0);
            }
            catch(...) {i -= 1;}
            i += 1;
        } while (i < 4);
    }
    else
    {
        for (int i(0); i<4; i++)
        {
            c.push_back(std::numeric_limits<double>::quiet_NaN());
            c.push_back(0);
        }
    }
    if (!initialization)
    {
        i = 0;
        do {
            std::cout << "Enter prev body twist command " << i << ": ";
            try {
                std::cin >> ui;
                c.push_back(ui);
            }
            catch(...) {i -= 1;}
            i += 1;
        } while (i < 2);
    }
    else
    {
        c.push_back(std::numeric_limits<double>::quiet_NaN());
        c.push_back(std::numeric_limits<double>::quiet_NaN());
    }
    DM control = DM(c);
    while (debug_mode)
    {
        i = 0;
        do {
            std::cout << "Enter command vector " << i << ": ";
            try {
                std::cin >> ui;
                control(2*i+1) = static_cast<DM>(ui);
            }
            catch(...) {i -= 1;}
            i += 1;
        } while (i < (nu-2) / 2);
        std::cout << "control vector: " << static_cast<std::vector<double>>(control) << std::endl;
        DM full_output = forward(state, control, true);
        DM output = full_output(Slice(0, nx));
        control(0) = (1 - r[0]) * control(0) + r[0] * control(1);
        control(2) = (1 - r[1]) * control(2) + r[1] * control(3);
        control(4) = (1 - r[2]) * control(4) + r[2] * control(5);
        control(6) = (1 - r[5]) * control(6) + r[5] * control(7);
        Eigen::Vector3d xy_command = forward_body_twist_command_w(state, control);   // Here u0(8) and u0(9) are zeros (useless).
        if (!std::isnan(static_cast<double>(control(8)))) {control(8) = r[3] * static_cast<DM>(xy_command(0)) + (1 - r[3]) * control(8);}
        else {control(8) = static_cast<DM>(xy_command(0));}
        if (!std::isnan(static_cast<double>(control(9)))) {control(9) = r[4] * static_cast<DM>(xy_command(1)) + (1 - r[4]) * control(9);}
        else {control(9) = static_cast<DM>(xy_command(1));}
        std::vector<double> vec = static_cast<std::vector<double>>(state);
        Eigen::Vector3d tba = forward_body_twist_w(vec[9], vec[12], vec[15], vec[10], vec[13], vec[16]);
        output(nx-2) = tba(0);
        output(nx-1) = tba(1);
        std::cout << "REAL OUTPUT: " << static_cast<std::vector<double>>(output) << std::endl;
        state = output;
        std::string sig;
        std::cout << "Press enter to continue ";
        std::cin >> sig;
    }
}





void MPC::build_solver()
{
    ROS_INFO("Building Solver...");

    SX X = SX::sym("X", nx, N+1);
    SX U = SX::sym("U", nu, N);
    SX X0 = SX::sym("X0", nx);
    SX U0 = SX::sym("U0", nu);
    //SX Xref = SX::sym("Xref", nx, N+1);
    SX Xref = SX::sym("Xref", nx*(N+1));
    SX cost = 0;
    std::vector<SX> constraints;
    constraints.push_back(X(Slice(), 0) - X0);
    
    for (int k = 0; k < N-1; k++) 
    {
        SX xk = X(Slice(), k);
        SX uk = U(Slice(), k);
        SX ukp1 = U(Slice(), k+1);
        SX xkp1 = X(Slice(), k+1);
        SX err = xk - Xref(Slice(k*nx, (k+1)*nx));
        cost += mtimes(mtimes(err.T(), Q), err);
        cost += mtimes(mtimes(uk.T(), R), uk);
        ROS_INFO("TEST");
        SX output = forward(xk, uk);
        constraints.push_back(xkp1 - output); // <------------------------------------------------------------------------------------------------------------------
    }
    ROS_INFO("Forward constraints built !");
    SX err = X(Slice(), N) - Xref(Slice((N-1)*nx, N*nx));
    cost += mtimes(mtimes(err.T(), P), err);
    
    // Recurrent constraints on inputs (1st order filter)
    SX acc = get_acceleration_commands(X0, U0);
    SX body_acc = toBody(X0, acc);
    constraints.push_back(U(Slice(), 0)(0) - r[0]*U0(1) - (1-r[0])*U0(0));        // U0 represents the current input
    constraints.push_back(U(Slice(), 0)(2) - r[1]*U0(3) - (1-r[1])*U0(2));
    constraints.push_back(U(Slice(), 0)(4) - r[2]*U0(5) - (1-r[2])*U0(4));
    constraints.push_back(U(Slice(), 0)(6) - r[5]*U0(7) - (1-r[5])*U0(6));
    constraints.push_back(U(Slice(), 0)(8) + r[3]*body_acc(1)/g - (1-r[3])*U0(8));            // Internal Differential Input
    constraints.push_back(U(Slice(), 0)(9) - r[4]*body_acc(0)/g - (1-r[4])*U0(9));            // Internal Differential Input
    for (int k = 0; k < N-1; k++)
    {
        SX xk(X(Slice(), k)), uk(U(Slice(), k));
        acc = get_acceleration_commands(xk, uk);
        body_acc = toBody(xk, acc);
        constraints.push_back(U(Slice(), k+1)(0) - r[0]*U(Slice(), k)(1) - (1-r[0])*U(Slice(), k)(0));
        constraints.push_back(U(Slice(), k+1)(2) - r[1]*U(Slice(), k)(3) - (1-r[1])*U(Slice(), k)(2));
        constraints.push_back(U(Slice(), k+1)(4) - r[2]*U(Slice(), k)(5) - (1-r[2])*U(Slice(), k)(4));
        constraints.push_back(U(Slice(), k+1)(6) - r[5]*U(Slice(), k)(7) - (1-r[5])*U(Slice(), k)(6));
        constraints.push_back(U(Slice(), k+1)(8) + r[3]*body_acc(1)/g - (1-r[3])*U(Slice(), k)(8));            // Internal Differential Input
        constraints.push_back(U(Slice(), k+1)(9) - r[4]*body_acc(0)/g - (1-r[4])*U(Slice(), k)(9));            // Internal Differential Input
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
    SX _g = vertcat(constraints);
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
        {"g", _g},
        {"p", SX::vertcat({X0, U0, Xref})},            // Symbols relative to non optimized parameters
    }, opts);
    
    // args for CALL
    args["lbg"] = DM::zeros(nx*(N+1) + (nu-4)*N);
    args["ubg"] = DM::zeros(nx*(N+1) + (nu-4)*N);
    args["lbx"] = DM(lbx);
    args["ubx"] = DM(ubx);
    //args["x0"] = DM::zeros(nlp.n_in(0));  // init guess
    //args["p"] = DM::vertcat({x0, xref(Slice(0, nx))}); // X0, Xref
    //args["p"] = DM::vertcat({x0, u0, xref(Slice(), Slice(0, N))}); // X0, U0, Xref  <----- params
    
    std::vector<double> sp((N+1)*nx + N*nu, 0.0);
    sol_prev = sp;
    
    init_x0_u0(true);
    
    xref_vec = reshape(xref, trajL*nx, 1);
    xref_vec_final = DM::zeros(nx*(N+1), 1);
    for (int i(0); i<N+1; i++)
    {
        xref_vec_final(i*nx)   = xref_vec(nx*trajL-nx);
        xref_vec_final(i*nx+3) = xref_vec(nx*trajL-nx+3);
        xref_vec_final(i*nx+6) = xref_vec(nx*trajL-nx+6);
        xref_vec_final(i*nx+9) = xref_vec(nx*trajL-nx+9);
        xref_vec_final(i*nx+12) = xref_vec(nx*trajL-nx+12);
        xref_vec_final(i*nx+15) = xref_vec(nx*trajL-nx+15);
    }
    
    std_msgs::MultiArrayDimension dim1, dim2;
    result_traj.layout.dim.push_back(dim1);
    result_traj.layout.dim.push_back(dim2);
    result_traj.layout.dim[0].label = "height";
    result_traj.layout.dim[1].label = "width";
    result_traj.layout.dim[0].size = trajL;
    result_traj.layout.dim[1].size = nx;
    result_traj.layout.dim[0].stride = trajL*nx;
    result_traj.layout.dim[1].stride = nx;
    
    ROS_INFO("SOLVER BUILT !");
    
    /*SX x = SX::sym("x", nx);
    SX Jg = jacobian(_g, x);
    Function Jg_fun = Function("Jg_fun", {x}, {Jg});
    DM Jg_val = Jg_fun(x0).at(0);
    std::cout << "Jacobian of g at x*: " << Jg_val << std::endl;*/
}


void build_linear_solver() {}


void MPC::iteration_test()
{
    ROS_INFO("Iteration test...");

    // Arbitrary init conditions
    init_x0_u0(true);
    ROS_INFO_STREAM("traj length: " << trajL);
    DM xref_vec = reshape(xref, nx*trajL, 1);
    std::vector<double> x0_guess((N+1)*nx + N*nu, 0.0);
    
    int test_step = 0;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    
    do
    {
        /*std::cout << "init state: " << x0(0) << " " << x0(1) << " " << x0(2) << " " << x0(3) << " " << x0(4) << " " << x0(5) << " " << x0(6) << " " << x0(7) << " " << x0(8) << " " << x0(9) << " " << x0(10) << " " << x0(11) << " " << x0(12) << " " << x0(13) << " " << x0(14) << " " << x0(15) << " " << x0(16) << " " << x0(17) << " " << x0(18) << " " << x0(19) << std::endl;
        std::cout << "init control: " << u0(0) << " " << u0(1) << " " << u0(2) << " " << u0(3) << " " << u0(4) << " " << u0(5) << " " << u0(6) << " " << u0(7) << " " << u0(8) << " " <<u0(9) << std::endl;*/
    
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
            //ROS_INFO_STREAM("x_opt_pred " << i+1 << ": " << std::endl << test_x_opt(nx*i) <<" "<< test_x_opt(nx*i+1) <<" "<< test_x_opt(nx*i+2) <<" "<< test_x_opt(nx*i+3) <<" "<< test_x_opt(nx*i+4) <<" "<< test_x_opt(nx*i+5) <<" "<< test_x_opt(nx*i+6) <<" "<< test_x_opt(nx*i+7) <<" "<< test_x_opt(nx*i+8) <<" "<< test_x_opt(nx*i+9) <<" "<< test_x_opt(nx*i+10) <<" "<< test_x_opt(nx*i+11));
        for (int i(0); i<N; i++)
            //ROS_INFO_STREAM("u_opt_pred " << i+1 << ": " << std::endl << test_u_opt(nu*i) <<" "<< test_u_opt(nu*i+1) <<" "<< test_u_opt(nu*i+2) <<" "<< test_u_opt(nu*i+3) <<" "<< test_u_opt(nu*i+4) <<" "<< test_u_opt(nu*i+5) <<" "<< test_u_opt(nu*i+6) <<" "<< test_u_opt(nu*i+7));
        
        x0 = test_x_opt(Slice(nx, 2*nx));          // As it is not simulated
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



void MPC::run_trajectory(const ros::TimerEvent&)
{   
    //auto now = std::chrono::steady_clock::now();
    
    init_x0_u0(false);
    
    for (int i(0); i<3; i++) {result_traj.data.push_back(static_cast<double>(x0(3*i)));}
    result_traj.data.push_back(current_euler(0));
    result_traj.data.push_back(current_euler(1));
    result_traj.data.push_back(current_euler(2));

    // Refining CALL args (Warm init)
    std::vector<double> x0_guess((N+1)*nx + N*nu, 0.0);
    for (int i = 0; i < N; i++) 
    {
        for (int j(0); j<nx; j++)
            x0_guess[i*nx + j] = sol_prev[(i+1)*nx + j];
        for (int j(0); j<nu; j++)
            x0_guess[(N+1)*nx + i*nu + j] = (i < N-1) ? sol_prev[(N+1)*nx + (i+1)*nu + j] : sol_prev[(N+1)*nx + (N-1)*nu];
    }
    for (int j(0); j<nx; j++)
        x0_guess[N*nx + j] = sol_prev[N*nx + j];
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
    input.linear.x = lx * cos(current_euler(2)) + ly * sin(current_euler(2));                  //Invert Rotmatrix
    input.linear.y = -lx * sin(current_euler(2)) + ly * cos(current_euler(2));
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
