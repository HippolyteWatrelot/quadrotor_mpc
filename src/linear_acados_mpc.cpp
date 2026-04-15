#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <map>
#include <string>

#include "quadrotor_mpc/linear_acados_mpc.h"
#include <casadi/casadi.hpp>
#include <limits>



linear_acados_MPC::linear_acados_MPC(bool gt, bool w)
{
    ground_truth = gt; 
    wind = w; 
    init_parameters();
}

linear_acados_MPC::~linear_acados_MPC() 
{
    for (int i(0); i<trajL; i++)
    {
        delete xref[i];
        delete uref[i];
    }
    delete xref_final;
    delete capsule;
}


void linear_acados_MPC::pose_callback(const geometry_msgs::PoseStamped::ConstPtr& pose)
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


void linear_acados_MPC::euler_callback(const geometry_msgs::Vector3Stamped::ConstPtr& vec)
{
    double prev_yaw = current_euler(2);
    double prev_yaw_vel = current_yaw_vel;
    current_euler(0)  = vec->vector.x;
    current_euler(1) = vec->vector.y;
    current_euler(2)   = vec->vector.z;
    current_yaw = current_euler(2);
    current_yaw_vel = (current_yaw - prev_yaw) / dt;
    current_yaw_acc = (current_yaw_vel - prev_yaw_vel) / dt;
}


void linear_acados_MPC::command_callback(const geometry_msgs::Twist::ConstPtr& twist)
{
    current_command(0) = twist->linear.x;
    current_command(1) = twist->linear.y;
    current_command(2) = twist->linear.z;
    current_command(3) = twist->angular.z;
}


void linear_acados_MPC::traj_length_callback(const std_msgs::Int16::ConstPtr& msg)
{
    trajL = msg->data;
    xref = std::vector<double*>(trajL, new double);
    for (int i(0); i<trajL; i++){for (int j(nx); j<nx+nu; j++) {*(xref[i]+j) = 0;}}
    ROS_INFO("traj length: %d", trajL);
}


void linear_acados_MPC::traj_points_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        *(xref[i]) = t->data[3*i];
        *(xref[i]+3) = t->data[3*i+1];
        *(xref[i]+6) = t->data[3*i+2];
    }
}


void linear_acados_MPC::traj_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        *(xref[i]+1) = t->data[3*i];
        *(xref[i]+4) = t->data[3*i+1];
        *(xref[i]+7) = t->data[3*i+2];
    }
}


void linear_acados_MPC::traj_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
    {
        *(xref[i]+2) = t->data[3*i];
        *(xref[i]+5) = t->data[3*i+1];
        *(xref[i]+7) = t->data[3*i+2];
    }
}


void linear_acados_MPC::traj_yaws_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        *(xref[i]+9) = t->data[i];
}


void linear_acados_MPC::traj_yaws_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        *(xref[i]+10) = t->data[i];
}


void linear_acados_MPC::traj_yaws_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t)
{
    for (int i(0); i<trajL; i++)
        *(xref[i]+11) = t->data[i];
}


void linear_acados_MPC::ui_command_callback(const std_msgs::Int16::ConstPtr& msg)
{
    if (msg->data==13 || msg->data==17) 
    {
        if (msg->data == 17)
            build_fixed_point();
        build_solver();
        timer = std::make_shared<ros::Timer>(_nh.createTimer(ros::Duration(delta_t), &linear_acados_MPC::run_trajectory, this));
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
                std::cout << " "<< *(xref[i]) <<" "<< *(xref[i]+1) <<" "<< *(xref[i]+2) <<" "<< *(xref[i]+3) <<" "<< *(xref[i]+4) <<" "<< *(xref[i]+5) <<" "<< *(xref[i]+6) <<" "<< *(xref[i]+7) <<" "<< *(xref[i]+8) <<" "<< *(xref[i]+9) <<" "<< *(xref[i]+10) << *(xref[i]+11) << std::endl;
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


void linear_acados_MPC::init_parameters() 
{   
    current_pose << 0, 0, 0;
    current_euler << 0, 0, 0;
    current_ori << 0, 0, 0, 0;
    //xref = Eigen::VectorXd::Zero(nx*N);
    //uref = Eigen::VectorXd::Zero(nu*N);
    
    pose_subscriber    = _nh.subscribe<geometry_msgs::PoseStamped>("/ground_truth_to_tf/pose", 1, &linear_acados_MPC::pose_callback, this);
    euler_subscriber   = _nh.subscribe<geometry_msgs::Vector3Stamped>("/ground_truth_to_tf/euler", 1, &linear_acados_MPC::euler_callback, this);
    command_subscriber = _nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 1, &linear_acados_MPC::command_callback, this);
    
    traj_length_subscriber = _nh.subscribe<std_msgs::Int16>("/trajectory_length", 1, &linear_acados_MPC::traj_length_callback, this);
    traj_points_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_points", 1, &linear_acados_MPC::traj_points_callback, this);
    traj_speeds_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_speeds", 1, &linear_acados_MPC::traj_speeds_callback, this);
    traj_accs_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_accs", 1, &linear_acados_MPC::traj_accs_callback, this);
    
    yaws_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_yaws", 1, &linear_acados_MPC::traj_yaws_callback, this);
    yaws_speeds_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_yaws_speeds", 1, &linear_acados_MPC::traj_yaws_speeds_callback, this);
    yaws_accs_subscriber = _nh.subscribe<std_msgs::Float64MultiArray>("/trajectory_yaws_accs", 1, &linear_acados_MPC::traj_yaws_accs_callback, this);
    
    ui_command_subscriber = _nh.subscribe<std_msgs::Int16>("/command", 1, &linear_acados_MPC::ui_command_callback, this);
    
    exe_pub = _nh.advertise<geometry_msgs::Twist>("/cmd_vel", 500, true);
    result_traj_pub = _nh.advertise<std_msgs::Float64MultiArray>("/result_poses", 1, true);
}



void linear_acados_MPC::build_fixed_point(bool test, double speed, int margin)
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
    //xref = std::vector<std::vector<double>>(nx, std::vector<double>(full_length));
    //uref = std::vector<std::vector<double>>(nu, std::vector<double>(full_length));
    xref = std::vector<double*>(full_length, new double);
    uref = std::vector<double*>(full_length, new double);
    trajL = full_length;
    for (int i(0); i<length; i++)
    {
        *(xref[i]) = pos(0) + i * (coords[0] - pos(0)) / length;
        *(xref[i]+3) = pos(1) + i * (coords[1] - pos(1)) / length;
        *(xref[i]+6) = pos(2) + i * (coords[2] - pos(2)) / length;
        *(xref[i]+9) = yaw + i * yaw_distance / length;
        
        *(xref[i]+1)  = (pos(0) - coords[0]) * speed / distance;
        *(xref[i]+4)  = (pos(1) - coords[1]) * speed / distance;
        *(xref[i]+7)  = (pos(2) - coords[2]) * speed / distance;
        *(xref[i]+10) = yaw_distance * speed / distance;
        
        *(xref[i]+2)  = 0;
        *(xref[i]+5)  = 0;
        *(xref[i]+8)  = 0;
        *(xref[i]+11) = 0;
    }
    for (int i(length); i<trajL; i++)
    {
        *(xref[i]) = coords[0];
        *(xref[i]+3) = coords[1];
        *(xref[i]+6) = coords[2];
        *(xref[i]+9) = yaw + yaw_distance;
        
        *(xref[i]+1)  = 0;
        *(xref[i]+4)  = 0;
        *(xref[i]+7)  = 0;
        *(xref[i]+10) = 0;
        
        *(xref[i]+2)  = 0;
        *(xref[i]+5)  = 0;
        *(xref[i]+8)  = 0;
        *(xref[i]+11) = 0;
    }
}



void linear_acados_MPC::build_solver()
{
    ROS_INFO("Building Solver...");
    
    status = drone_dynamics_acados_create(capsule);
    if (status != 0) {ROS_ERROR("Failed to create acados linear solver!");}
    else {ROS_INFO("LINEAR SOLVER CREATED !");}
    
    // Init guess
    std::fill(x0, x0+nx, 0);
    x0[6] = 0.2;
    
    x0_guess_run = std::vector<std::vector<double>>(N+1, std::vector<double>(nx, 0)); 
    u0_guess_run = std::vector<std::vector<double>>(N, std::vector<double>(nu, 0)); ;
    
    u0[0] = current_command(0)*cos(current_yaw) - current_command(1)*sin(current_yaw);
    u0[1] = current_command(0)*sin(current_yaw) + current_command(1)*cos(current_yaw);
    u0[2] = current_command(2);
    u0[3] = current_command(3);
    
    xref_final = new double;
    *xref_final     = *(xref[trajL-1]);
    *(xref_final+3) = *(xref[trajL-1]+3);
    *(xref_final+6) = *(xref[trajL-1]+6);
    *(xref_final+9) = *(xref[trajL-1]+9);
    
    std_msgs::MultiArrayDimension dim1, dim2;
    result_traj.layout.dim.push_back(dim1);
    result_traj.layout.dim.push_back(dim2);
    result_traj.layout.dim[0].label = "height";
    result_traj.layout.dim[1].label = "width";
    result_traj.layout.dim[0].size = trajL;
    result_traj.layout.dim[1].size = 12;
    result_traj.layout.dim[0].stride = trajL*12;
    result_traj.layout.dim[1].stride = 12;
}


void linear_acados_MPC::iteration_test()
{
    ROS_INFO("Iteration test...");

    // Arbitrary init conditions
    build_solver();
    
    ROS_INFO_STREAM("traj length: " << trajL);
    //std::vector<std::vector<double>> ux_guess0(N+1, std::vector<double>(nx+nu), 0.0);
    
    std::vector<std::vector<double>> x0_guess(N+1, std::vector<double>(nx)), u0_guess(N, std::vector<double>(nu));
    
    double time_steps[N];
    for(int i(0); i<N; i++) {time_steps[i] = i+1;}
    drone_dynamics_acados_update_time_steps(capsule, N, time_steps);
    
    int test_step = 0;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    
    // Initial conditions
    ocp_nlp_constraints_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, 0, "lbx", x0);
    ocp_nlp_constraints_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, 0, "ubx", x0);
    
    // First guess
    double ux0[nx+nu];
    for (int i(0); i<N; i++) 
    {
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "x", x0);
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "u", u0);
    }
    
    do
    {
        // Sliding Horizon on xref
        try 
        {
            // SET REFERENCE
            double yref[nx];
            for (int i(0); i<N; i++)
            {
                memcpy(yref, xref[i+test_step], nx * sizeof(double));
                ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "yref", yref);
            }
            memcpy(yref, xref[N+test_step], nx * sizeof(double));
            ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N, "yref_e", yref);
        }
        catch(...) {break;}
        
        /*std::cout << "init state: " << x0[0] << " " << x0[1] << " " << x0[2] << " " << x0[3] << " " << x0[4] << " " << x0[5] << " " << x0[6] << " " << x0[7] << " " << x0[8] << " " << x0[9] << " " << x0[10] << " " << x0[11] << std::endl;
        std::cout << "init control: " << u0[0] << " " << u0[1] << " " << u0[2] << " " << u0[3] << " " << u0[4] << " " << u0[5] << " " << u0[6] << " " << u0[7] << std::endl;*/
        
        // RESOLUTION
        status = drone_dynamics_acados_solve(capsule);
        // NEW INIT STATE AND CONTROL
        ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, 0, "x", &x0);
        ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, 0, "u", &u0);
        
        // Warm guess
        ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, 0, "u", &(u0_guess[0]));
        for (int i(1); i<N+1; i++) 
        {
            ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, i, "x", &(x0_guess[i-1]));
            ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, i, "u", &(u0_guess[i]));
            ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "x", &(x0_guess[i]));
            ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "u", &(u0_guess[i]));
        }
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, 0, "x", &(x0_guess[0]));
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, 0, "u", &(u0_guess[0]));
        
        /*for (int i(0); i<N; i++)
            ROS_INFO_STREAM("x_opt_pred " << i+1 << ": " << std::endl << x0_guess[i][nu] <<" "<< x0_guess[i][nu+1] <<" "<< x0_guess[i][nu+2] <<" "<< x0_guess[i][nu+3] <<" "<< x0_guess[i][nu+4] <<" "<< x0_guess[i][nu+5] <<" "<< x0_guess[i][nu+6] <<" "<< x0_guess[i][nu+7] <<" "<< x0_guess[i][nu+8] <<" "<< x0_guess[i][nu+9] <<" "<< x0_guess[i][nu+10] <<" "<< x0_guess[i][nu+11]);
        for (int i(0); i<N; i++)
            ROS_INFO_STREAM("u_opt_pred " << i+1 << ": " << std::endl << u0_guess[i][0] <<" "<< u0_guess[i][1] <<" "<< u0_guess[i][2] <<" "<< u0_guess[i][3]);*/
        
        //std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000*delta_t)));
        test_step++;
        
    } while (true);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    ROS_INFO_STREAM("Run time = " << elapsed_seconds.count() << "s\n");
    ROS_INFO("Test ends !");
    
    drone_dynamics_acados_free(capsule);
    drone_dynamics_acados_free_capsule(capsule);
    delete capsule;
}



void linear_acados_MPC::run_trajectory(const ros::TimerEvent&)
{   
    //auto now = std::chrono::steady_clock::now();
    
    x0[0] = current_pose(0);
    x0[1] = current_vel(0);
    x0[2] = current_acc(0);
    x0[3] = current_pose(1);
    x0[4] = current_vel(1);
    x0[5] = current_acc(1);
    x0[6] = current_pose(2);
    x0[7] = current_vel(2);
    x0[8] = current_acc(2);
    x0[9] = current_yaw;
    x0[10] = current_yaw_vel;
    x0[11] = current_yaw_acc;
    
    // Collect data
    for (int i(0); i<3; i++) {result_traj.data.push_back(x0[3*i]);}
    result_traj.data.push_back(current_euler(0));
    result_traj.data.push_back(current_euler(1));
    result_traj.data.push_back(x0[9]);

    try 
    {
        // SET REFERENCE and WARM GUESS
        memcpy(yref, xref[N+step], (nx+nu) * sizeof(double));
        ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N, "yref_e", yref);
        memcpy(yref, xref[N-1+step], (nx+nu) * sizeof(double));
        ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N-1, "yref", yref);
        for (int i(0); i<N-1; i++)
        {
            memcpy(yref, xref[i+step], (nx+nu) * sizeof(double));
            ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "yref", yref);
            
            ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, i+1, "x", &(x0_guess_run[i]));
            ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, i+1, "u", &(u0_guess_run[i]));
            ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "x", &(x0_guess_run[i]));
            ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "u", &(u0_guess_run[i]));
        }
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N-1, "x", &(x0_guess_run[N-1]));
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N, "x", &(x0_guess_run[N]));
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N-1, "u", &(u0_guess_run[N-1]));
    }
    catch(...)  // End of trajectory
    {
        memcpy(yref, xref_final, (nx+nu) * sizeof(double));
        ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N, "yref_e", yref);
        ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N-1, "yref", yref);
        for (int i(0); i<N-1; i++)
        {
            ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "yref", yref);
        
            ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, i+1, "x", &(x0_guess_run[i]));
            ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, i+1, "u", &(u0_guess_run[i]));
            ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "x", &(x0_guess_run[i]));
            ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "u", &(u0_guess_run[i]));
        }
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N-1, "x", &(x0_guess_run[N-1]));
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N, "x", &(x0_guess_run[N]));
        ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N-1, "u", &(u0_guess_run[N-1]));
    }
    
    // CALL
    status = drone_dynamics_acados_solve(capsule);
    // NEW INIT STATE AND CONTROL
    ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, 0, "x", &x0);
    ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, 0, "u", &u0);
        
    // PUBLISHING IN STABLE FRAME
    input.linear.x = u0[0] * cos(current_yaw) + u0[1] * sin(current_yaw);                  //Invert Rotmatrix
    input.linear.y = -u0[0] * sin(current_yaw) + u0[1] * cos(current_yaw);
    input.linear.z = u0[2];
    input.angular.x = 0;
    input.angular.y = 0;
    input.angular.z = u0[3];
    exe_pub.publish(input);  // <--------------------------------------------------------------------------------
    
    step++;
}
