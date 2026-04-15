#! /usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rospy
import sys
import os

from quadrotor_mpc import transform_utils as tu
#from transform_utils import euler_to_quaternion, pose_to_transmatrix, TransInv, MatrixLog6, se3ToVec

from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Int16


nx = 12
ref_traj = None
result_traj = None
errors = None


def traj_length_callback(msg):
    global length, ref_traj, result_traj, errors
    length = msg.data
    ref_traj = np.zeros([length, 6])
    result_traj = np.zeros([length, 6])
    errors = np.zeros(length)


def reference_points_callback(tab):
    global length, ref_traj, nx
    for i in range(length):
        ref_traj[i, :3] = tab.data[3*i:3*(i+1)]
        
        
def reference_yaws_callback(tab):
    global length, ref_traj, nx
    for i in range(length):
        ref_traj[i, 5] = tab.data[i]



def results_callback(tab):
    global length, result_traj, nx
    for i in range(length):
        result_traj[i, :] = tab.data[i*6:(i+1)*6]
        
        
        
def command_callback(msg):
    if msg.data == 20:
        errors_positions()
        
        

def errors_positions():
    global length, ref_traj, result_traj
    #try:
    Xerrs = []
    for i in range(length):
        ref_euler, result_euler = ref_traj[i, 2:].tolist(), result_traj[i, 2:].tolist()
        ref_quat, result_quat = tu.euler_to_quaternion(ref_euler[0], ref_euler[1], ref_euler[2]), tu.euler_to_quaternion(result_euler[0], result_euler[1], result_euler[2])
        ref_transmat, result_transmat = tu.pose_to_transmatrix(ref_traj[i, :3].tolist(), ref_quat), tu.pose_to_transmatrix(result_traj[i, :3].tolist(), result_quat)
        Mat_Xerr = tu.MatrixLog6(tu.TransInv(result_transmat) @ ref_transmat)
        Xerr = tu.se3ToVec(Mat_Xerr)
        Xerrs.append(Xerr.tolist())
    plt.plot(Xerrs)
    plt.savefig("/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/screens/Xerrs.png")
    #plt.show()
    #except:
        #print('NO RESULT TRAJECTORY')
    


if __name__ == "__main__":
    rospy.init_node('data')
    command_sub = rospy.Subscriber('/command', Int16, command_callback)
    traj_length_sub = rospy.Subscriber('/trajectory_length', Int16, traj_length_callback)
    ref_points_sub = rospy.Subscriber('/trajectory_points', Float64MultiArray, reference_points_callback)
    ref_yaws_sub = rospy.Subscriber('/trajectory_yaws', Float64MultiArray, reference_yaws_callback)
    result_sub = rospy.Subscriber('/result_poses', Float64MultiArray, results_callback)
    rospy.spin()
