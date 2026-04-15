cmake_minimum_required(VERSION 3.0.2)
project(quadrotor_mpc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(cmake_modules)
IF(cmake_modules_FOUND)
  find_package(Eigen3 REQUIRED)
ELSE()
  find_package(Eigen3 REQUIRED)
  #set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
ENDIF()

#set(CASADI_DIR "$ENV{HOME}/opt/casadi-3.7.0")
#include_directories(${CASADI_DIR}/include)
#link_directories(${CASADI_DIR}/lib)

find_package(CasADi REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ibex REQUIRED ibex)
#pkg_check_modules(dlib REQUIRED dlib)
#find_package(dlib REQUIRED)
#target_link_libraries( ${CUR_PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} dlib::dlib)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  eigen_conversions
  message_generation
)

add_service_files(
  FILES
  Equilibrium.srv
  Robust_Equilibrium.srv
  Tube.srv
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES acados_ocp_solver_drone_dynamics
  LIBRARIES acados_ocp_solver_full_drone_dynamics
  CATKIN_DEPENDS roscpp rospy
  CATKIN_DEPENDS std_msgs
  CATKIN_DEPENDS message_runtime
)


catkin_install_python(PROGRAMS scripts/records.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/plot_robot_path.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/MPC_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/results.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/Acados_Generators/generate_linear_ocp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/Acados_Generators/generate_full_ocp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/msg_pyfiles/Classic_Invariant_msg.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/Invariant_Design/nodes/Classic_LJ_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/tube_validation_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  .
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${CASADI_INCLUDE_DIRS}
  ${ACADOS_INSTALL_DIR}/include
  ${ACADOS_INSTALL_DIR}/interfaces/acados_template/include
  ${ACADOS_INSTALL_DIR}/external/include
  #${PROJECT_SOURCE_DIR}/src/acados_solver/c_generated_code
  ${PROJECT_SOURCE_DIR}/include/quadrotor_mpc/acados_solver/linear
  ${PROJECT_SOURCE_DIR}/include/quadrotor_mpc/acados_solver/full
  #${qpOASES_INCLUDE_DIRS}
  #qpOASES
  #casadi
  ${ibex_INCLUDE_DIRS}
)

link_directories(
  ${ACADOS_INSTALL_DIR}/lib
  ${PROJECT_SOURCE_DIR}/include/quadrotor_mpc/acados_solver/linear
  ${PROJECT_SOURCE_DIR}/include/quadrotor_mpc/acados_solver/full
)

#file(GLOB DYN_FUN_SOURCES
#  "${PROJECT_SOURCE_DIR}/c_generated_code/drone_dynamics/*.c"
#)

#add_library(acados_ocp_solver_drone_dynamics SHARED
#  ${DYN_FUN_SOURCES}
#)

#add_executable(MPC_node 
#  src/MPC_node.cpp
#  src/MPC2.cpp
#  )
#target_link_libraries(MPC_node ${catkin_LIBRARIES})
#target_link_libraries(MPC_node ${catkin_LIBRARIES} qpOASES)
#add_dependencies(MPC_node quadrotor_mpc_gencpp)

add_executable(example_gen_ac src/example_gen_ac.cpp)
target_link_libraries(example_gen_ac ${catkin_LIBRARIES} casadi)
#target_link_libraries(casadi_model_node ${catkin_LIBRARIES} ${CASADI_LIBRARIES})
add_dependencies(example_gen_ac quadrotor_mpc_gencpp)



#add_executable(linear_acados_model_node 
#  src/Acados/linear/linear_acados_mpc_node.cpp
#  src/Acados/linear/linear_acados_mpc.cpp
#  )
#target_link_libraries(linear_acados_model_node 
#  ${catkin_LIBRARIES}
#  acados_ocp_solver_drone_dynamics
#  acados
#  hpipm
#  blasfeo
#)
#add_dependencies(linear_acados_model_node quadrotor_mpc_gencpp)

#add_executable(full_acados_model_node 
#  src/Acados/full/full_acados_mpc_node.cpp
#  src/Acados/full/full_acados_mpc.cpp
#  )
#target_link_libraries(full_acados_model_node 
#  ${catkin_LIBRARIES}
#  acados_ocp_solver_full_drone_dynamics
#  acados
#  hpipm
#  blasfeo
#)
#add_dependencies(full_acados_model_node quadrotor_mpc_gencpp)



add_executable(linear_mpc_node 
  src/linear_mpc_node.cpp
  src/linear_mpc.cpp
  )
target_link_libraries(linear_mpc_node ${catkin_LIBRARIES} casadi)
add_dependencies(linear_mpc_node quadrotor_mpc_gencpp)

add_executable(full_mpc_node 
  src/full_mpc_node.cpp
  src/full_mpc.cpp
  )
target_link_libraries(full_mpc_node ${catkin_LIBRARIES} casadi ${ibex_LIBRARIES})
add_dependencies(full_mpc_node quadrotor_mpc_gencpp)

#add_executable(acados_mpc_node src/acados_mpc_node.cpp)
#target_link_libraries(acados_mpc_node ${catkin_LIBRARIES} acados)
