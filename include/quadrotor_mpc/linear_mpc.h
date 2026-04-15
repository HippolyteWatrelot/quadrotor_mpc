#ifndef LINEAR_MPC_H_INCLUDED
#define LINEAR_MPC_H_INCLUDED

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


class linear_MPC
{
public:
    linear_MPC(bool gt, bool w);
    ~linear_MPC();
    Eigen::Vector3d quaternion_to_euler(Eigen::Vector4d& q) const;
    std::vector<int> T;
    std::vector<int> control_dim;
    casadi::SX A, B, C, D, K;
    casadi::SX Q, R, P;
    casadi::SX NQ, NR;
    casadi::DM Mx, Mc;
    std::vector<double> Kvec;
    Eigen::MatrixXd Gamma, H, Phi;
    void cost_matrices();
    void N_cost_matrices();
    static Eigen::MatrixXd matrix_power(Eigen::MatrixXd M, int n);
    void build_fixed_point(bool test=false, double speed=0.5, int margin=100);
    void build_solver();
    void iteration_test();
    void run_trajectory(const ros::TimerEvent&);

protected:
    void command_callback(const geometry_msgs::Twist::ConstPtr& twist);
    void pose_callback(const geometry_msgs::PoseStamped::ConstPtr& pose);
    void euler_callback(const geometry_msgs::PoseStamped::ConstPtr& pose);
    void traj_length_callback(const std_msgs::Int16::ConstPtr& msg);
    void traj_points_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_yaws_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_yaws_speeds_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void traj_yaws_accs_callback(const std_msgs::Float64MultiArray::ConstPtr& t);
    void timestep_callback(const std_msgs::Float32::ConstPtr& dt);
    void ui_command_callback(const std_msgs::Int16::ConstPtr& msg);
    void init_parameters();
    void build_ABCD();
    void constraints_matrices();
    //static double cost(const Eigen::VectorXd& c, const Eigen::VectorXd& x, const Eigen::VectorXd& xref, const Eigen::VectorXd& uref);

private:
    bool ground_truth;
    bool wind;
    std::string method;
    int trajL=0; 
    int N;
    int nx=12; 
    int nu=8;
    int step = 0;
    double P_factor;
    std::vector<double> sol_prev;
    casadi::Function solver;
    std::map<std::string, casadi::DM> args;
    casadi::DM x0, u0;
    casadi::DM u_opt;
    std_msgs::Float64MultiArray result_traj;
    geometry_msgs::Twist input;
    Eigen::Vector3d current_pose, current_vel, current_acc, current_euler;
    double current_yaw, current_yaw_vel, current_yaw_acc;
    Eigen::Vector4d current_ori, current_command;
    casadi::DM xref, uref;
    casadi::DM xref_vec, xref_vec_final;
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
    ros::Subscriber pose_subscriber, euler_subscriber, command_subscriber;
    ros::Subscriber traj_length_subscriber, traj_points_subscriber, traj_speeds_subscriber, traj_accs_subscriber;
    ros::Subscriber yaws_subscriber, yaws_speeds_subscriber, yaws_accs_subscriber;
    ros::Subscriber timestep_subscriber, ui_command_subscriber;
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

#endif // LINEAR_MPC_H_INCLUDED
