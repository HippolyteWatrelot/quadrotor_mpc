#include <ros/ros.h>
//#include "quadrotor_mpc/MPC.h"
#include "quadrotor_mpc/full_acados_mpc.h"


int main(int argc, char *argv[])
{
  ros::init(argc, argv, "full_mpc_node");

  full_acados_MPC mpc(true, false);
  //mpc.run_trajectory();

  ros::spin();

  return 0;
}

