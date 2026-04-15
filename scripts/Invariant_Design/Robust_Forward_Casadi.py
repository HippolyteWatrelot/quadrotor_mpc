import numpy as np
from math import atan2
import matplotlib.pyplot as plt
from casadi import *
import yaml
import sys
from quadrotor_mpc.transform_utils import euler_to_quaternion, VecToso3, euler_to_rotmatrix, eulerd2w, wd2eulerdd, cross



nx = 9
nu = 4
g = 9.8065

I = np.array([[0.0115202,         0,         0],
              [        0, 0.0115457,         0],
              [        0,         0, 0.0218256]])
I1, I2, I3 = I[0, 0], I[1, 1], I[2, 2]
mass = 1.478
mass_offset = 0.05
Interval = {'I00':  [I1 - I1*mass_offset/mass, I1 + I1*mass_offset/mass], 
            'I11':  [I2 - I2*mass_offset/mass, I2 + I2*mass_offset/mass], 
            'I22':  [I3 - I3*mass_offset/mass, I3 + I3*mass_offset/mass], 
            'mass': [mass-mass_offset, mass+mass_offset]}
gamma_xy, gamma_z, gamma_wx, gamma_wy, gamma_wz = 5.1, 5.1, 10.5, 10.5, 5.25
r = [2/3, 2/3, 2/3, 10/11, 10/11, 0.5]
use_Interval = False

d_xq, d_yq, d_z, roll, rolld, pitch, pitchd, yawd = MX.sym('d_xq'), MX.sym('d_yq'), MX.sym('d_z'), MX.sym('roll'), MX.sym('rolld'), MX.sym('pitch'), MX.sym('pitchd'), MX.sym('yawd')                                                             
ux, uy, uz, uwz = MX.sym('ux'), MX.sym('uy'), MX.sym('uz'), MX.sym('uwz')
wx, wy, wz = MX.sym('wx'), MX.sym('wy'), MX.sym('wz') 


def stable_frame(state, control):
    _yaw = 0   #state[7]
    real_control = MX([0, 0, 0, 0])
    real_control[0] = cos(_yaw) * control[0] - sin(_yaw) * control[1]
    real_control[1] = sin(_yaw) * control[0] + cos(_yaw) * control[1]
    real_control[2] = control[2]
    real_control[3] = control[3]
    return real_control


def toWorld(x):
    new_x = [None for _ in range(8)]
    # positions are useless
    yaw = 0   #x[7]
    new_x[0] = x[0]*cos(yaw) - x[1]*sin(yaw)
    new_x[1] = x[0]*sin(yaw) + x[1]*cos(yaw)
    new_x[2] = x[2]
    for i in range(3, 8):
        new_x[i] = x[i]
    return new_x
    
    
def get_acceleration_commands(state, command):
    global gamma_x, gamma_y, gamma_z, r, g
    speeds = state[:3]
    command_ux = command[0]
    acceleration_command_x = -gamma_xy * speeds[0] + gamma_xy * command_ux
    command_uy = command[1]
    acceleration_command_y = -gamma_xy * speeds[1] + gamma_xy * command_uy
    command_uz = command[2]
    acceleration_command_z = -gamma_z * speeds[2] + gamma_z * command_uz + g
    return [acceleration_command_x, acceleration_command_y, acceleration_command_z]
    
    
def load_factor(euler_angles):
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
    #q = euler_to_quaternion(yaw, pitch, roll)
    return 1 / (cos(roll)*cos(pitch))
    
    
def get_force(state, acceleration_command_z):
    global g, gamma_z, r
    euler_angles = state[3], state[5], 0
    return mass * ((acceleration_command_z - g) * load_factor(euler_angles) + g)
    
    
def toBody(state, vec):
    euler_angles = state[3], state[5], 0
    m = euler_to_rotmatrix(euler_angles[2], euler_angles[1], euler_angles[0])
    return np.dot(m.T, vec)
    
    
def get_torques(acceleration_command, state, command, rel_force):
    global I, gamma_wx, gamma_wy, gamma_wz, g, r
    acceleration_command_body = toBody(state, acceleration_command)                          # body frame
    uwx_body, uwy_body = -acceleration_command_body[1]/g, acceleration_command_body[0]/g
    command_uwz = command[3]
    wz = eulerd2w([state[3], state[5], 0], [state[4], state[6], state[7]])[2]
    torque_x = I[0, 0] * gamma_wx * uwx_body
    torque_y = I[1, 1] * gamma_wy * uwy_body
    torque_z = I[2, 2] * gamma_wz * (-wz + command_uwz)
    torques = [torque_x, torque_y, torque_z]
    return torques
    
    
def get_euler_accs(state, torques):
    global mass, I
    euler = state[3], state[5], 0
    eulerd = state[4], state[6], state[7]
    speeds = state[0], state[1], state[2]
    R = euler_to_rotmatrix(euler[2], euler[1], euler[0])
    wb = eulerd2w(euler, eulerd)
    body_twist = list(wb) + list(R.T @ np.array(speeds))     # linear in speeds
    
    d_body_twist = []
    d_body_twist.append((torques[0] + body_twist[1]*body_twist[2]*(I[2, 2] - I[1, 1]))/I[0, 0])
    d_body_twist.append((torques[1] + body_twist[2]*body_twist[0]*(I[0, 0] - I[2, 2]))/I[1, 1])
    d_body_twist.append((torques[2] + body_twist[0]*body_twist[1]*(I[1, 1] - I[0, 0]))/I[2, 2])
    d_body_twist.append(body_twist[1]*body_twist[5] - body_twist[2]*body_twist[4])
    d_body_twist.append(body_twist[2]*body_twist[3] - body_twist[0]*body_twist[5])
    d_body_twist.append(body_twist[0]*body_twist[4] - body_twist[1]*body_twist[3])
    
    eulerdd = wd2eulerdd(euler, eulerd, np.array(d_body_twist)[:3])
    return eulerdd
    
    
def get_cartesian_accs(state, rel_force_command, rel_noise):
    global g, mass
    roll, pitch, yaw = state[3], state[5], 0
    accs = np.array([rel_force_command/mass * (np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)) + noise[0]/mass,
                     rel_force_command/mass * (np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)) + noise[1]/mass,
                     rel_force_command/mass  * np.cos(pitch)*np.cos(roll) + noise[2]/mass - g])
    return [accs[0], accs[1], accs[2]]
    
    
def local(state, output):
    yaw = 0   #state[7]
    eulerd = [state[4], state[6], state[7]]
    d2xq = output[0]*cos(yaw) + output[1]*sin(yaw)
    d2yq = -output[0]*sin(yaw) + output[1]*cos(yaw)
    output[:2] = d2xq, d2yq
    final_output = output[:3] + [eulerd[0]] + [output[3]] + [eulerd[1]] + [output[4]] + [output[5]]
    print(final_output)
    return vertcat(*final_output)


def forward(raw_x, u, w):
    x = toWorld(raw_x)
    real_u = stable_frame(x, u)
    acc_commands = get_acceleration_commands(x, real_u)
    relative_z_force = get_force(x, acc_commands[2])
    planar_torques = get_torques(acc_commands, x, real_u, relative_z_force)
    output_cart_acc = get_cartesian_accs(x, relative_z_force, w)
    output_euler_acc = get_euler_accs(x, planar_torques)
    output_acc = list(output_cart_acc) + list(output_euler_acc)
    local_output_acc = local(x, output_acc)
    return local_output_acc
    
    
if __name__ == "__main__":
    input_x = vertcat(d_xq, d_yq, d_z, roll, rolld, pitch, pitchd, yawd)
    input_u = vertcat(ux, uy, uz, uwz)
    input_w = vertcat(wx, wy, wz)
    if bool(int(sys.argv[1])):
        I1, I2, I3 = MX.sym('I00'), MX.sym('I11'), MX.sym('I22')
        input_y = vertcat(I1, I2, I3)
        output = forward(input_x, input_u, input_w)
        f = Function('f', [input_x, input_u, input_w, input_y], [output])
        f.save('CasADi_formulas/Robust_dyn_f_INTERVAL.casadi')
    else:
        output = forward(input_x, input_u, input_w)
        f = Function('f', [input_x, input_u, input_w], [output])
        f.save('CasADi_formulas/Robust_dyn_f.casadi')
