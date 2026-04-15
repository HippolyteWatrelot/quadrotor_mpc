# quadrotor_mpc

Classic MPC on the Hector Quadrotor Robot Model
Point & Trajectory tracking

Simulation
<img src="screens/test3.webm">

## Gettin started
### Install ROS

Project developed on ROS Noetic, Ubuntu 20.04

**Install screen**

```
sudo apt install screen
```

**Install YAD**
Get instructions on the following link:
https://doc.ubuntu-fr.org/yad_yet_another_dialog


**Create a workspace**

```
mkdir -p quadrotor_ws/src
cd quadrotor_ws/src && catkin_init_workspace
```

**To be used with Hector Quadrotor**

Clone quadrotor repo in your workspace

```
git clone https://github.com/RAFALAMAO/hector-quadrotor-noetic.git
cd ~/path3D_ws && catkin build
```

**Clone repository then build wokspace**

```
git clone https://gitlab.ensta.fr/ssh/quadrotor_mpc.git
cd ~/quadrotor_ws && catkin build
```

## Use

```
cd ~/quadrotor_mpc
screen -c screenrc_procedure_test
```
Navigate through the screens with Ctrl+a+n or Ctrl+a+p
Create new screen with Ctrl+a+c
Exit screen by typing "exit"
