#ifndef MPC_H_INCLUDED
#define MPC_H_INCLUDED

#include <map>
#include <vector>
#include <cmath>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <assert.h>
#include <numeric>
#include <random>
#include <iterator>
#include <ros/ros.h>
#include <ros/node_handle.h>
#include <XmlRpcValue.h>

#include "std_msgs/Int16.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float64MultiArray.h"

#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "geometry_msgs/Twist.h"


class MPC
{
public:
    MPC(bool gt, bool w);
    ~MPC();
    std::vector<int> T;
    std::vector<int> control_dim;
    Eigen::MatrixXd A, B, C, D;
    Eigen::MatrixXd Q, R;
    Eigen::MatrixXd NQ, NR;
    Eigen::MatrixXd Mx, Mc;
    std::vector<double> K;
    Eigen::MatrixXd Gamma, H, Phi;
    void cost_matrices();
    void N_cost_matrices();
    static Eigen::MatrixXd matrix_power(Eigen::MatrixXd M, int n);
    void run_trajectory();

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
    void timestep_callback(const std_msgs::Float32::ConstPtr& dt);
    void ui_command_callback(const std_msgs::Int16::ConstPtr& msg);
    void init_parameters();
    void build_ABCD();
    void get_mats();
    void constraints_matrices();
    Eigen::Vector3d toBody(const Eigen::Vector3d& space);
    Eigen::Vector3d fromBody(const Eigen::Vector3d& body);
    void printMatrix(const std::string& name, Eigen::MatrixXd mat, int rows, int cols);
    //static double cost(const Eigen::VectorXd& c, const Eigen::VectorXd& x, const Eigen::VectorXd& xref, const Eigen::VectorXd& uref);

private:
    bool ground_truth;
    bool wind;
    int trajL=0; 
    int N;
    int nx=12; 
    int nu=8;
    Eigen::Vector3d current_pose, current_vel, current_acc, current_euler;
    double current_yaw_vel, current_yaw_acc;
    Eigen::Vector4d current_ori, current_command;
    Eigen::VectorXd xref, uref;
    double delta_t;
    double dt=0.01;
    double lxy_kp, lxy_ki, lxy_kd;
    double lz_kp, lz_ki, lz_kd;
    double az_kp, az_ki, az_kd;
    double lxy_tau, lz_tau, az_tau;
    std::vector<double> umin, umax;
    std::vector<Eigen::Vector3d> traj_points, traj_speeds, traj_accs;
    std::vector<double> traj_yaws, traj_yaws_speeds, traj_yaws_accs;
    ros::NodeHandle _nh;
    ros::Subscriber pose_subscriber, euler_subscriber, command_subscriber;
    ros::Subscriber traj_length_subscriber, traj_points_subscriber, traj_speeds_subscriber, traj_accs_subscriber;
    ros::Subscriber yaws_subscriber, yaws_speeds_subscriber, yaws_accs_subscriber;
    ros::Subscriber timestep_subscriber, ui_command_subscriber;
    ros::Publisher exe_pub;
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

#endif // MPC_H_INCLUDED
