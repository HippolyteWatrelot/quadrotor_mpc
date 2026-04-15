import numpy as np
from math import atan2
import matplotlib.pyplot as plt


nx = 20
nu = 10
I = np.array([[0.0115202,         0,         0],
              [        0, 0.0115457,         0],
              [        0,         0, 0.0218256]])
I1, I2, I3 = I[0, 0], I[1, 1], I[2, 2]
g = 9.8065
mass = 1.478
#CoG = [-0.000108, 0, -8e-6]



period = 0.1

lxy_kp = 5
lxy_ki = 1
lxy_tau = 0.05
    
lz_kp = 5
lz_ki = 1
lz_tau = 0.05
    
axy_kp = 10
axy_ki = 5
axy_kd = 5
    
az_kp = 5
az_ki = 2.5
    
axy_tau = 0.01
az_tau = 0.1


gamma_x, gamma_y, gamma_z, gamma_wx, gamma_wy, gamma_wz = [lxy_kp + lxy_ki*period, lxy_kp + lxy_ki*period, lz_kp + lz_ki*period,
                                                           axy_kp + axy_ki*period, axy_kp + axy_ki*period, az_kp + az_ki*period]
#r = [period / (period + lxy_tau), period / (period + lxy_tau), period / (period + lz_tau), period / (period + axy_tau), period / (period + axy_tau), period / (period + az_tau)]




def VecToso3(omg):
    return np.arrray([[0,      -omg[2],  omg[1]],
                      [omg[2],       0, -omg[0]],
                      [-omg[1], omg[0],       0]])
                     
                     
def eulerd2w(euler, eulerd):
    roll, pitch, yaw = euler
    m = np.array([[1,          0,          -sin(pitch)], 
                  [0,  cos(roll), cos(pitch)*sin(roll)], 
                  [0, -sin(roll), cos(pitch)*cos(roll)]])
    return m @ np.array(eulerd)
    
    
def wd2eulerdd(euler, eulerd, wd):
    roll, pitch, yaw = euler
    rolld, pitchd, yawd = eulerd
    minv = np.array(([[1,    np.tan(pitch)*sin(roll),    np.tan(pitch)*cos(roll)], 
                      [0,               np.cos(roll),              -np.sin(roll)], 
                      [0, np.sin(roll)/np.cos(pitch), np.cos(roll)/np.cos(pitch)]])
    md = np.array([[0,                   0,                                                 -pitchd*np.cos(pitch)], 
                   [0, -rolld*np.sin(roll),  rolld*np.cos(pitch)*np.cos(roll) - pitchd*np.sin(roll)*np.sin(pitch)], 
                   [0, -rolld*np.cos(roll), -rolld*np.cos(pitch)*np.sin(roll) - pitchd*np.cos(roll)*np.sin(pitch)]])
    return minv @ (np.array(wd) - md @ np.array(eulerd))
    
    
def eulerdd2wd(euler, eulerd, eulerdd):
    roll, pitch, yaw = euler
    rolld, pitchd, yawd = eulerd
    m = np.array([[1,          0,          -sin(pitch)], 
                  [0,  cos(roll), cos(pitch)*sin(roll)], 
                  [0, -sin(roll), cos(pitch)*cos(roll)]])
    md = np.array([[0,                   0,                                                 -pitchd*np.cos(pitch)], 
                   [0, -rolld*np.sin(roll),  rolld*np.cos(pitch)*np.cos(roll) - pitchd*np.sin(roll)*np.sin(pitch)], 
                   [0, -rolld*np.cos(roll), -rolld*np.cos(pitch)*np.sin(roll) - pitchd*np.cos(roll)*np.sin(pitch)]])
    return m @ eulerdd + md @ eulerd


def toState(state, vec_body):
    roll, pitch, yaw = state[9], state[12], state[15]
    m = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.cos(roll)*np.sin(pitch) + np.sin(yaw)*np.sin(roll)], 
                  [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.cos(roll)*np.sin(pitch) - np.cos(yaw)*np.sin(roll)], 
                  [           -np.sin(pitch),                                        np.sin(roll)*np.cos(pitch),                                        np.cos(roll)*np.cos(pitch)]])
    vec = m @ vec_body
    return vec
    
    
def toBody(state, vec):
    roll, pitch, yaw = state[9], state[12], state[15]
    m = Matrix([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll)], 
                [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll)], 
                [        -sin(pitch),                               sin(roll)*cos(pitch),                               cos(roll)*cos(pitch)]])
    vec_body = m.transpose() * vec
    return vec_body
    
    
def AdjointTwist(twist):
    v, w = twist[:3], twist[3:]
    AT = np.zeros([6, 6])
    AT[:3, :3] = VecToso3(w);
    AT[:3, 3:] = np.zeros([3, 3])
    AT[3:, :3] = VecToso3(v)
    AT[3:, 3:] = VecToso3(w)
    return AT

    
def Adjoint(state):
    pos = state[:9:3]
    roll, pitch, yaw = state[9], state[12], state[15] 
    Ad = np.zeros([6, 6])
    R = euler_to_rotmatrix(yaw, pitch, roll)
    Ad[:3, :3] = R
    Ad[:3, 3:] = np.zeros([3, 3])
    Ad[3:, :3] = VecToso3(pos) * R
    Ad[3:, 3:] = R
    return Ad
    
def AdjointInvert(state):
    pos = state[:9:3]
    roll, pitch, yaw = state[9], state[12], state[15] 
    AdI = np.zeros([6, 6])
    R = euler_to_rotmatrix(yaw, pitch, roll)
    AdI[:3, :3] = -R.T
    AdI[:3, 3:] = np.zeros([3, 3])
    AdI[3:, :3] = -R.T * VecToso3(pos)
    AdI[3:, 3:] = -R.T
    return AdI




def rfc(acc_z, state):
    global g, mass
    roll, pitch = state[9], state[12]
    force = mass*(acc_z + g) / (np.cos(pitch)*np.cos(roll))
    return force


def torques_from_euler_accs(eulerdd, state):
    global I1, I2, I3
    euler, eulerd = state[9:16:3], state[10:17:3]
    R = euler_to_rotmatrix(euler[2], euler[1], euler[0])
    body_twist = np.array(eulerd2w(euler, eulerd).tolist() + list(R.T @ np.array([state[1], state[4], state[7]])))
    d_body_twist_angular = eulerdd2wd(euler, eulerd, eulerdd)
    # twist_state 3 and 4 and twist_body 3 and 4 are supposed to be 0
    torque_x = I1 * d_body_twist_angular[0] - body_twist[1]*body_twist[2]*(I3 - I2))  # should be 0 if linear
    torque_y = I2 * d_body_twist_angular[1] - body_twist[2]*body_twist[0]*(I1 - I3))  # should be 0 if linear
    torque_z = I3 * d_body_twist_angular[2] - body_twist[0]*body_twist[1]*(I2 - I1))
    return [torque_x, torque_y, torque_z]


def uwz_from_torque(tz, state):
    global I3, gamma_z
    wz = eulerd2w([state[9], state[12], state[15]], [state[10], state[13], state[16]])[2]
    uwz = tz / (I3 * gamma_wz) + wz
    return uwz


def command_acz_from_force(rfc, state):
    global g, mass
    euler_angles = np.array([state[9], state[12], state[15]])
    lf = load_factor(euler_angles)
    acz = (rfc/mass - g) / lf + g
    return acz


def command_acxy_from_torques(torques, acc_command_z, state):
    global I1, I2, gamma_x, gamma_y, g
    uwx_body = torques[0] / (I1 * gamma_x)
    uwy_body = torques[1] / (I2 * gamma_y)
    acc_body_x =  g * uwy_body
    acc_body_y = -g * uwx_body
    acc_body_z = (acc_command_z + acc_body_x*np.sin(pitch) - acc_body_y*np.sin(roll)*cos(pitch)) / (np.cos(roll)*np.cos(pitch))
    acc_command = toState(state, [acc_body_x, acc_body_y, acc_body_z])
    return acc_command[0], acc_command[1]


def inputs_from_accs_commands(ax, ay, az, state):
    global g
    prev_vx, prev_vy, prev_vz = state[1:9:3]
    ux = (gamma_x*prev_vx + ax) / gamma_x
    uy = (gamma_y*prev_vy + ay) / gamma_y
    uz = (gamma_z*prev_vz + az - g) / gamma_z
    return ux, uy, uz
    
    
def previous_state(state):
    global dt
    roll, pitch = state[9], state[12]
    x, xd, y, yd, z, zd, roll, pitch, yaw, yawd = state[0], state[1], state[3], state[4], state[6], state[7], state[9], state[12], state[15], state[16]
    prev_yaw = yaw - yawd*dt
    prev_xd, prev_yd = xd*np.cos(prev_yaw - yaw) - yd*np.sin(prev_yaw - yaw), xd*np.sin(prev_yaw - yaw) + yd*np.cos(prev_yaw - yaw)
    prev_xdd, prev_ydd = -yawd * (np.sin(prev_yaw)*prev_xd + np.cos(prev_yaw)*prev_yd), yawd * (np.cos(prev_yaw)*prev_xd - np.sin(prev_yaw)*prev_yd)
    prev_x = x - prev_xd*dt - prev_xdd*(dt**2)/2
    prev_y = y - prev_yd*dt - prev_ydd*(dt**2)/2
    prev_z = z - zd*dt
    return [prev_x, prev_xd, prev_xdd, prev_y, prev_yd, prev_ydd, prev_z, zd, 0, roll, 0, 0, pitch, 0, 0, prev_yaw, yawd, 0]


def control_backward(values, interval_mode=False):
    state = [values['x'], values['xd'], values['xdd'], values['y'], values['yd'], values['ydd'], values['z'], values['zd'], values['zdd'], values['roll'], 0, 0, values['pitch'], 0, 0, values['yaw'], values['yawd'], 0]
    if interval_mode:
        I1, I2, I3, mass = Symbol('I00'), Symbol('I11'), Symbol('I22'), Symbol('mass')
    else:
        I1, I2, I3 = I[0, 0], I[1, 1], I[2, 2]
    prev_state = previous_state(state)
    cart_accs, euler_accs = state[2:9:3], state[11:18:3]   # xdd, ydd = yawd*(-sin(yaw)*input_twist[0] - cos(yaw)*input_twist[1]), yawd*(cos(yaw)*input_twist[0] - sin(yaw)*input_twist[1])
    rel_force_command = rfc(cart_accs, prev_state)
    torques = torques_from_euler_accs(euler_accs, prev_state)
    acz = command_acz_from_force(rel_force_command, prev_state)
    acx, acy = command_acxywz_from_torques(torques)
    ux, uy, uz = inputs_from_accs_commands(acx, acy, acz, prev_state)
    uwz = uwz_from_torque(torques[2], prev_state)
    return [ux, uy, uz, uwz]
