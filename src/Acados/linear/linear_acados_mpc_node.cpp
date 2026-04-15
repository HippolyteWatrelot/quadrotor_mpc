#include <ros/ros.h>
//#include "quadrotor_mpc/MPC.h"
#include "quadrotor_mpc/linear_acados_mpc.h"


int main(int argc, char *argv[])
{
  ros::init(argc, argv, "linear_mpc_node");

  linear_acados_MPC mpc(true, false);
  //mpc.run_trajectory();

  ros::spin();

  return 0;
}

