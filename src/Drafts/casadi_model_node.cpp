#include <ros/ros.h>
#include "quadrotor_mpc/casadi_model.h"


int main(int argc, char *argv[])
{
  ros::init(argc, argv, "mpc_node");

  casadi_model cm(true, false);
  int n(std::accumulate(cm.T.begin(), cm.T.end(), 0));
  
  std::cout << std::endl << std::endl;;
  std::cout << "A: " << std::endl;
  for (int i(0); i<n; i++)
  {
      for (int j(0); j<n; j++)
          std::cout << cm.A(i, j) << " ";
      std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;;
  std::cout << "B: " << std::endl;
  for (int i(0); i<n; i++)
  {
      for (int j(0); j<8; j++)
          std::cout << cm.B(i, j) << " ";
      std::cout << std::endl;
  }
  
  std::cout << std::endl << std::endl;;
  std::cout << "Q: " << std::endl;
  for (int i(0); i<n; i++)
  {
      for (int j(0); j<12; j++)
          std::cout << cm.Q(i, j) << " ";
      std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;;
  std::cout << "R: " << std::endl;
  for (int i(0); i<8; i++)
  {
      for (int j(0); j<8; j++)
          std::cout << cm.R(i, j) << " ";
      std::cout << std::endl;
  }

  cm.build_model();

  return 0;
}

