#! /usr/bin/env python3

import numpy as np
import random
import rospy
from std_msgs.msg import Int16
from geometry_msgs.msg import Vector3
import os
import sys

cf = os.path.abspath(os.getcwd())
wind = dict()
wind_topic = None
sd = {'x': 0.15, 'y': 0.15, 'z': 0.15}
rv = dict()
wind_trigger = 21
state = False



def command_callback(msg):
    global state
    if msg.data == wind_trigger:
        state == True
        set_wind()
    elif msg.data == wind_trigger_off:
        state == False


def parameters_init():
    global sd, wind, rv
    for _, axis in enumerate(['x', 'y', 'z']):
        rv[axis] = random.gauss(wind[axis], sd[axis])


def set_wind():
    global wind, sd, state
    if state == 1:
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and state:
            msg = Vector3()
            msg.x = random.gauss(wind['x'], sd['x']*wind['x'])
            msg.y = random.gauss(wind['y'], sd['y']*wind['y'])
            msg.z = random.gauss(wind['z'], sd['z']*wind['z'])
            wind_pub.publish(msg)
            rate.sleep()
        


if __name__ == "__main__":
    rospy.init_node("wind_setup")
    command_sub = rospy.Subscriber('/command', Int16, command_callback)
    wind_pub = rospy.Publisher('/wind', Vector3, queue_size=100)
    for _, axis in enumerate(['x', 'y', 'z']):
        while True:
            try:
                wind[axis] = float(input(f'wind force on {axis}: '))
                break
            except:
                pass
    parameters_init()
    state = True
    set_wind()
    rospy.spin()
