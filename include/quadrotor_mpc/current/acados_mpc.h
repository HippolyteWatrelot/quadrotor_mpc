#ifndef ACADOS_MPC_H_INCLUDED
#define ACADOS_MPC_H_INCLUDED

#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <assert.h>
#include <numeric>
#include <random>
#include <iterator>
#include <ros/ros.h>
#include <ros/node_handle.h>

#include <casadi/casadi.hpp>

#include "std_msgs/Int16.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/MultiArrayDimension.h"

#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "geometry_msgs/Twist.h"

//#include "acados_solver_drone_dynamics.h"
//#include "drone_dynamics_model.h"

extern "C" {
    #include "acados_solver/full/acados_solver_drone_dynamics.h"
    #include "acados_solver/full/drone_dynamics_model/drone_dynamics_model.h"
}


class acados_MPC
{
public:
    acados_MPC(bool gt, bool w);
    ~acados_MPC();
    std::vector<int> control_dim;
    void build_fixed_point(bool test=false, double speed=0.5, int margin=100);
    void build_solver();
    void iteration_test();
    void run_trajectory(const ros::TimerEvent&);

protected:
    void command_callback(const geometry_msgs::Twist::ConstPtr& twist);
    void pose_callback(const geometry_msgs::PoseStamped::ConstPtr& pose);
    void euler_callback(const geometry_msgs::Vector3Stamped::ConstPtr& vec);
    void traj_length_callback(const std_msgs::Int16::ConstPtr& msg);
    void traj_points_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_yaws_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_yaws_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_yaws_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void ui_command_callback(const std_msgs::Int16::ConstPtr& msg);
    void init_parameters();
    //static double cost(const Eigen::VectorXd& c, const Eigen::VectorXd& x, const Eigen::VectorXd& xref, const Eigen::VectorXd& uref);

private:
    drone_dynamics_solver_capsule *capsule;    // struct
    int status;
    bool ground_truth;
    bool wind;
    std::string method;
    int trajL=0; 
    int N;
    static const int nx=16;
    static const int state_nx=12; 
    static const int nu=4;
    int step = 0;
    double P_factor;
    double x0[nx], u0[nu], yref[nx+nu];
    std::vector<std::vector<double>> x0_guess_run, u0_guess_run;
    double u_opt[nu];
    std_msgs::Float64MultiArray result_traj;
    geometry_msgs::Twist input;
    Eigen::Vector3d current_pose, current_vel, current_acc, current_euler;
    double current_yaw, current_yaw_vel, current_yaw_acc;
    Eigen::Vector4d current_ori, current_command;
    std::vector<double*> xref, uref;
    double* xref_final;
    double delta_t;
    double dt=0.01;
    double lxy_kp, lxy_ki, lxy_kd;
    double lz_kp, lz_ki, lz_kd;
    double az_kp, az_ki, az_kd;
    double lxy_tau, lz_tau, az_tau;
    std::vector<double> xmin, xmax;
    std::vector<double> umin, umax;
    std::vector<Eigen::Vector3d> traj_points, traj_speeds, traj_accs;
    std::vector<double> traj_yaws, traj_yaws_speeds, traj_yaws_accs;
    ros::NodeHandle _nh;
    ros::Subscriber pose_subscriber, euler_subscriber, command_subscriber, ui_command_subscriber;
    ros::Subscriber traj_length_subscriber, traj_points_subscriber, traj_speeds_subscriber, traj_accs_subscriber;
    ros::Subscriber yaws_subscriber, yaws_speeds_subscriber, yaws_accs_subscriber;
    ros::Publisher exe_pub, result_traj_pub;
    std::shared_ptr<ros::Timer> timer;
    std::vector<double> yaws;
    std::vector<double> yaws_speeds;
    std::vector<double> yaws_accs;
    ros::Subscriber traj_sub, speeds_sub, accs_sub, yaws_sub, yaws_speeds_sub, yaws_accs_sub;
    struct state {
      Eigen::Vector3d position;
      Eigen::Vector3d euler_angles;
      double inertia_coeffs[6];
      Eigen::Vector3d linear_speeds;
      Eigen::Vector3d angular_speeds;
      Eigen::Vector3d linear_accelerations;
      Eigen::Vector3d angular_accelerations;
    } s;
};

#endif // ACADOS_MPC_H_INCLUDED
