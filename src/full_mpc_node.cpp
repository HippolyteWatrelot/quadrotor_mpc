#include <ros/ros.h>
//#include "quadrotor_mpc/MPC.h"
#include "quadrotor_mpc/full_mpc.h"


int main(int argc, char *argv[])
{
  ros::init(argc, argv, "mpc_node");

  MPC mpc(true, false);
  
  std::cout << std::endl << std::endl;
  std::cout << "Q: " << std::endl;
  for (int i(0); i<20; i++)
  {
      for (int j(0); j<20; j++)
          std::cout << mpc.Q(i, j) << " ";
      std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;;
  std::cout << "R: " << std::endl;
  for (int i(0); i<10; i++)
  {
      for (int j(0); j<10; j++)
          std::cout << mpc.R(i, j) << " ";
      std::cout << std::endl;
  }
  
  //mpc.run_trajectory();

  ros::spin();

  return 0;
}

