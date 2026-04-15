#! /usr/bin/env python3

import numpy as np
import random
import rospy
import sys
import os

from std_msgs.msg import Float32MultiArray, Float32, MultiArrayDimension, Int16
from geometry_msgs.msg import Point, Vector3, PoseStamped


rec = False
path = []


def command_callback(msg):
    global rec
    global path
    if msg.data == 1:
        rec = True
    elif msg.data == 2:
        rec = False
        cf = os.path.abspath(os.getcwd())
        with open(cf + '/src/mpc_demos/scripts/npy/path.npy', 'wb') as f:
            np.save(f, np.array(path))


def path_callback(p):
    global rec
    global path
    if rec:
        point = list(np.zeros(3))
        point[0] = p.pose.position.x
        point[1] = p.pose.position.y
        point[2] = p.pose.position.z
        path.append(point)


if __name__ == "__main__":
    rospy.init_node("records")
    command_sub = rospy.Subscriber("/command", Int16, command_callback)
    path_sub = rospy.Subscriber("/ground_truth_to_tf/pose", PoseStamped, path_callback)
    rospy.spin()
