#include <ros/ros.h>
//#include "quadrotor_mpc/MPC.h"
#include "quadrotor_mpc/linear_mpc.h"


int main(int argc, char *argv[])
{
  ros::init(argc, argv, "mpc_node");

  linear_MPC mpc(true, false);
  int n(std::accumulate(mpc.T.begin(), mpc.T.end(), 0));
  
  std::cout << std::endl << std::endl;
  std::cout << "A: " << std::endl;
  for (int i(0); i<n; i++)
  {
      for (int j(0); j<n; j++)
          std::cout << mpc.A(i, j) << " ";
      std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;
  std::cout << "B: " << std::endl;
  for (int i(0); i<n; i++)
  {
      for (int j(0); j<8; j++)
          std::cout << mpc.B(i, j) << " ";
      std::cout << std::endl;
  }
  
  std::cout << std::endl << std::endl;
  std::cout << "Q: " << std::endl;
  for (int i(0); i<n; i++)
  {
      for (int j(0); j<12; j++)
          std::cout << mpc.Q(i, j) << " ";
      std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;
  std::cout << "R: " << std::endl;
  for (int i(0); i<8; i++)
  {
      for (int j(0); j<8; j++)
          std::cout << mpc.R(i, j) << " ";
      std::cout << std::endl;
  }
  
  //mpc.run_trajectory();

  ros::spin();

  return 0;
}

