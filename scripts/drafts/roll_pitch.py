#! /usr/bin/env python3

import numpy as np
import random
import rospy
import sys
import os
from sympy import Symbol, sin, cos, tan, atan, Matrix, Transpose, Derivative, simplify, srepr, factor, trigsimp, solve


nx = 20
nu = 10
I = Matrix([[Symbol('I00'), 0, 0],
            [0, Symbol('I11'), 0],
            [0, 0, Symbol('I22')]])
g = Symbol('g')
mass = Symbol('mass')
gamma_x, gamma_y, gamma_z, gamma_wx, gamma_wy, gamma_wz = Symbol('gamma_x'), Symbol('gamma_y'), Symbol('gamma_z'), Symbol('gamma_wx'), Symbol('gamma_wy'), Symbol('gamma_wz')


# rolld and pitchd are 0, we have to solve the system getting rolldd and pitchdd = 0
roll, rolld, rolldd = Symbol('roll'), 0, 0
pitch, pitchd, pitchdd = Symbol('pitch'), 0, 0
yaw, yawd, yawdd = Symbol('yaw'), Symbol('yawd'), 0




def VecToso3(omg):
    return Matrix([[0,      -omg[2],  omg[1]],
                   [omg[2],       0, -omg[0]],
                   [-omg[1], omg[0],       0]])
                   
                   
def euler_to_rotmatrix(yaw, pitch, roll):
    return Matrix([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll)], 
                   [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll)], 
                   [        -sin(pitch),                               sin(roll)*cos(pitch),                              cos(roll)*cos(pitch)]])
                   
                   
def eulerd2w(euler, eulerd):
    roll, pitch, yaw = euler
    m = Matrix([[1,          0,          -sin(pitch)], 
                [0,  cos(roll), cos(pitch)*sin(roll)], 
                [0, -sin(roll), cos(pitch)*cos(roll)]])
    return m * Matrix(eulerd)
    
    
def wd2eulerdd(euler, eulerd, wd):
    roll, pitch, yaw = euler
    rolld, pitchd, yawd = eulerd
    minv = Matrix([[1, tan(pitch)*sin(roll), tan(pitch)*cos(roll)], 
                   [0,            cos(roll),           -sin(roll)], 
                   [0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)]])
    md = Matrix([[0,                0,                                        -pitchd*cos(pitch)], 
                 [0, -rolld*sin(roll),  rolld*cos(pitch)*cos(roll) - pitchd*sin(roll)*sin(pitch)], 
                 [0, -rolld*cos(roll), -rolld*cos(pitch)*sin(roll) - pitchd*cos(roll)*sin(pitch)]])
    return minv * (Matrix(wd) - md * Matrix(eulerd))


def AdjointTwist(twist):
    v, w = twist[:3], twist[3:]
    AT = Matrix.zeros(6, 6)
    AT[:3, :3] = VecToso3(w);
    AT[:3, 3:] = Matrix.zeros(3, 3)
    AT[3:, :3] = VecToso3(v)
    AT[3:, 3:] = VecToso3(w)
    return AT

def Adjoint(state):
    pos = state[:9:3]
    roll, pitch, yaw = state[9], state[12], state[15] 
    Ad = Matrix.zeros(6, 6)
    R = euler_to_rotmatrix(yaw, pitch, roll)
    Ad[:3, :3] = R
    Ad[:3, 3:] = Matrix.zeros(3, 3)
    Ad[3:, :3] = VecToso3(pos) * R
    Ad[3:, 3:] = R
    return Ad
    
def AdjointInvert(state):
    pos = state[:9:3]
    roll, pitch, yaw = state[9], state[12], state[15] 
    AdI = Matrix.zeros(6, 6)
    R = euler_to_rotmatrix(yaw, pitch, roll)
    AdI[:3, :3] = -R.transpose()
    AdI[:3, 3:] = Matrix.zeros(3, 3)
    AdI[3:, :3] = -R.transpose() * VecToso3(pos)
    AdI[3:, 3:] = -R.transpose()
    return AdI


def toBody(state, vec):
    roll, pitch, yaw = state[9], state[12], state[15]
    m = Matrix([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll)], 
                [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll)], 
                [        -sin(pitch),                               sin(roll)*cos(pitch),                               cos(roll)*cos(pitch)]])
    vec_body = m.transpose() * vec
    return vec_body




def load_factor(euler_angles):
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
    return 1 / (cos(roll)*cos(pitch))
    
    
def get_force(state, acceleration_command_z):
    global g, mass
    euler_angles = Matrix([state[9], state[12], state[15]])
    lf = load_factor(euler_angles)
    return mass * ((acceleration_command_z - g) * lf + g);


def get_acceleration_commands(state, command):
    global gamma_x, gamma_y, gamma_z, r, g
    cart_speeds = [state[1], state[4], state[7]]
    ux = command[1]
    acceleration_command_x = -gamma_x * cart_speeds[0] + gamma_x * ux
    uy = command[3]
    acceleration_command_y = -gamma_y * cart_speeds[1] + gamma_y * uy
    uz = command[5]
    acceleration_command_z = -gamma_z * cart_speeds[2] + gamma_z * uz + g
    #acceleration_command_z = g * (2 - cos(roll)*cos(pitch))
    return Matrix([acceleration_command_x, acceleration_command_y, acceleration_command_z])


def get_torques(acceleration_command, twist_space, state, rel_force):
    global I, gamma_wx, gamma_wy, gamma_wz, g
    acceleration_command_body = toBody(state, acceleration_command)
    uwx_body, uwy_body = -acceleration_command_body[1]/g, acceleration_command_body[0]/g
    uwz = twist_space[5]
    wz = eulerd2w([state[9], state[12], state[15]], [state[10], state[13], state[16]])[2]
    torque_x = I[0, 0] * gamma_wx * uwx_body
    torque_y = I[1, 1] * gamma_wy * uwy_body
    torque_z = I[2, 2] * gamma_wz * (-wz + uwz)
    torques = Matrix([torque_x, torque_y, torque_z])
    return torques


def get_euler_accs(state, torques, twist_space):
    global mass, I
    euler = state[9:16:3]
    eulerd = state[10:17:3]
    speeds = []
    twist_body = AdjointInvert(state) * twist_space
    
    d_twist_body = []
    d_twist_body.append(twist_body[4]*twist_body[2] - twist_body[5]*twist_body[1])
    d_twist_body.append(twist_body[5]*twist_body[0] - twist_body[3]*twist_body[2])
    d_twist_body.append(twist_body[3]*twist_body[1] - twist_body[4]*twist_body[0])
    d_twist_body.append((torques[0] + twist_body[4]*twist_body[5]*(I[2, 2] - I[1, 1]))/I[0, 0])
    d_twist_body.append((torques[1] + twist_body[5]*twist_body[3]*(I[0, 0] - I[2, 2]))/I[1, 1])
    d_twist_body.append((torques[2] + twist_body[3]*twist_body[4]*(I[1, 1] - I[0, 0]))/I[2, 2])
    d_twist_body = Matrix(d_twist_body)
    
    d_twist_space = AdjointTwist(twist_space) * twist_space + Adjoint(state) * d_twist_body
    eulerdd = wd2eulerdd(euler, eulerd, d_twist_space[3:])
    return eulerdd
    
    
def get_cartesian_accs(state, rel_force_command):
    global mass, g
    roll, pitch, yaw = state[9], state[12], state[15]
    acc_x = simplify(rel_force_command/mass * (cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll)))
    acc_y = simplify(rel_force_command/mass * (sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll)))
    acc_z = simplify(rel_force_command/mass * cos(pitch)*cos(roll) - g)
    return np.array([acc_x, acc_y, acc_z])


def rp_solve(input_twist):

    global roll, pitch, yaw, yawd
    
    x, xd, xdd = 0, cos(yaw)*input_twist[0] - sin(yaw)*input_twist[1], yawd*(-sin(yaw)*input_twist[0] - cos(yaw)*input_twist[1])
    y, yd, ydd = 0, sin(yaw)*input_twist[0] + cos(yaw)*input_twist[1], yawd*(cos(yaw)*input_twist[0] - sin(yaw)*input_twist[1])
    z, zd, zdd = 0, input_twist[2], 0
    
    uz = (g*(1-cos(roll)*cos(pitch)) + gamma_z) / gamma_z
    pitch = -atan(input_twist[1]/g)
    roll = -atan(input_twist[1]*cos(pitch)/g)
    
    state = [0, xd, xdd, 0, yd, ydd, 0, zd, zdd, roll, 0, 0, pitch, 0, 0, yaw, yawd, 0]
    command = [Symbol('ux'), Symbol('uy'), uz, 0, 0, Symbol('uwz')]

    twist_space = Matrix([xd, yd, zd, 0, 0, yawd])
    twist_body = AdjointInvert(state) * twist_space
    tba = twist_body[3:]
    state = state + [tba[0], tba[1]]
    
    acc_commands = get_acceleration_commands(state, command)
    relative_z_force = get_force(state, acc_commands[2])
    torques = get_torques(acc_commands, twist_space, state, relative_z_force)
    cart_acc = get_cartesian_accs(state, relative_z_force)
    eulerdd = get_euler_accs(state, torques, twist_space)
    return eulerdd, cart_acc[2]
        
        
if __name__ == "__main__":

    desired_twist = Matrix([Symbol('xd'), Symbol('yd'), uz, 0, 0, Symbol('wzd')])
    eulerdd, acc_z = rp_solve(desired_twist)
    for i in range(3):
        print(eulerdd[i], '\n')
    print(acc_z)
    pitch = Symbol('pitch')
    #roll, pitch = Symbol('roll'), Symbol('pitch')
    p = solve([eulerdd[0], eulerdd[1], eulerdd[2]], pitch)
    #print('r', r)
    print('p', p)
