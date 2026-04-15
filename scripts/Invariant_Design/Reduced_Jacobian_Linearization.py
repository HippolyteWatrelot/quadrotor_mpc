import numpy as np
from math import atan2
import matplotlib.pyplot as plt
#from casadi import *
from sympy import *
import yaml
import sys



g = 9.8065
dt = 0.01

I = np.array([[0.0115202,         0,         0],
              [        0, 0.0115457,         0],
              [        0,         0, 0.0218256]])
I1, I2, I3 = Symbol('I1'), Symbol('I2'), Symbol('I3')
mass = Symbol("mass")
_mass = 1.478
mass_offset = 0.05
Interval = {'I00':  [I[0, 0] - I[0, 0]*mass_offset/_mass, I[0, 0] + I[0, 0]*mass_offset/_mass], 
            'I11':  [I[1, 1] - I[1, 1]*mass_offset/_mass, I[1, 1] + I[1, 1]*mass_offset/_mass], 
            'I22':  [I[2, 2] - I[2, 2]*mass_offset/_mass, I[2, 2] + I[2, 2]*mass_offset/_mass], 
            'mass': [_mass-mass_offset, _mass+mass_offset]}
#gamma_xy, gamma_z, gamma_wx, gamma_wy, gamma_wz = 5.1, 5.1, 10.5, 10.5, 5.25
lxy_kp, lz_kp, axy_kp, az_kp = 5, 5, 10, 5
lxy_ki, lz_ki, axy_ki, az_ki = 1, 1, 5, 2.5
#lxy_ki, lz_ki, axy_ki, az_ki = 0, 0, 0, 0
axy_kd = 5
#r = [2/3, 2/3, 2/3, 10/11, 10/11, 0.5]
#r = [1/6, 1/6, 1/6, 1/2, 1/2, 1/11]
use_Interval = False

d_xq, d_yq, d_z, roll, rolld, pitch, pitchd, yawd = Symbol('d_xq'), Symbol('d_yq'), Symbol('d_z'), Symbol('roll'), Symbol('rolld'), Symbol('pitch'), Symbol('pitchd'), Symbol('yawd')
d2_xq, d2_yq, d2_z, rolldd, pitchdd, yawdd = Symbol('d2_xq'), Symbol('d2_yq'), Symbol('d2_z'), Symbol('rolldd'), Symbol('pitchdd'), Symbol('yawdd')
prev_uwx_b, prev_uwy_b = Symbol('prev_uwx_b'), Symbol('prev_uwy_b')
d_prev_uwx_b, d_prev_uwy_b = Symbol('d_prev_uwx_b'), Symbol('d_prev_uwy_b')
                                                                                                                 
ux, uy, uz, uwz = Symbol('ux'), Symbol('uy'), Symbol('uz'), Symbol('uwz')

trigger = True

C_mxy = 0.0741562
C_mz = 0.0506433
C_wxy = 0.12
C_wz = 0.1

#prev_uwx_body, prev_uwy_body = 0, 0





def VecToso3(omg):
    return Matrix([[0,      -omg[2],  omg[1]],
                   [omg[2],       0, -omg[0]],
                   [-omg[1], omg[0],       0]])
                     
                     
def eulerd2w(euler, eulerd):
    roll, pitch = euler[:2]
    m = Matrix([[1,          0,          -sin(pitch)], 
                [0,  cos(roll), cos(pitch)*sin(roll)], 
                [0, -sin(roll), cos(pitch)*cos(roll)]])
    return m * Matrix(eulerd)
    
    
def wd2eulerdd(euler, eulerd, wd):
    roll, pitch = euler[:2]
    rolld, pitchd = eulerd[:2]
    minv = Matrix([[1, tan(pitch)*sin(roll), tan(pitch)*cos(roll)], 
                   [0,            cos(roll),           -sin(roll)], 
                   [0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)]])
    md = Matrix([[0,                0,                                        -pitchd*cos(pitch)], 
                 [0, -rolld*sin(roll),  rolld*cos(pitch)*cos(roll) - pitchd*sin(roll)*sin(pitch)], 
                 [0, -rolld*cos(roll), -rolld*cos(pitch)*sin(roll) - pitchd*cos(roll)*sin(pitch)]])
    return minv * (Matrix(wd) - md * Matrix(eulerd))
  
                     
def euler_to_quaternion(yaw, pitch, roll):
    qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
    qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
    qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)
    qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)
    return [qw, qx, qy, qz]


def euler_to_rotmatrix(yaw, pitch, roll):
    return Matrix([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*cos(roll)*sin(pitch) + sin(yaw)*sin(roll)], 
                   [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*cos(roll)*sin(pitch) - cos(yaw)*sin(roll)], 
                   [        -sin(pitch),                               sin(roll)*cos(pitch),                              cos(roll)*cos(pitch)]])






def stable_frame(x, u):
    yaw = 0 # x[7]   #state[7] useless
    real_u = [0, 0, 0, 0]
    real_u[0] = cos(yaw) * u[0] - sin(yaw) * u[1]
    real_u[1] = sin(yaw) * u[0] + cos(yaw) * u[1]
    real_u[2] = u[2]
    real_u[3] = u[3]
    return real_u


def toWorld(x):
    new_x = [None for _ in range(10)]
    # positions are useless
    roll, pitch, yaw = x[3], x[5], 0 # x[7]
    R = euler_to_rotmatrix(yaw, pitch, roll)
    X = R * Matrix(x[:3])
    new_x[:3] = X[0], X[1], X[2]
    #new_x[0] = x[0]*np.cos(yaw) - x[1]*np.sin(yaw)
    #new_x[1] = x[0]*np.sin(yaw) + x[1]*np.cos(yaw)
    for i in range(2, len(new_x)):
        new_x[i] = x[i]
    return new_x
    
    
def get_acceleration_commands(state, command):
    global lxy_kp, lz_kp, lxy_ki, lz_ki, gamma_xy, gamma_z, r, g, dt
    v = state[:3]
    #error_x, error_y, error_z = state[11:14]
    command_ux = command[0]
    acceleration_command_x = lxy_kp * (command_ux - v[0]) # + lxy_ki * error_x
    command_uy = command[1]
    acceleration_command_y = lxy_kp * (command_uy - v[1]) # + lxy_ki * error_y
    command_uz = command[2]
    acceleration_command_z = lz_kp * (command_uz - v[2]) + g # + lz_ki * error_z
    acceleration_command = [acceleration_command_x, acceleration_command_y, acceleration_command_z]
    d_error = [command_ux - v[0], command_uy - v[1], command_uz - v[2]]
    return acceleration_command, d_error
    
    
def load_factor(roll, pitch):
    return 1 / (cos(roll)*cos(pitch))
    
    
def get_force_z(state, acbz):
    global g, gamma_z, r
    lf = load_factor(state[3], state[5])
    return mass * ((acbz - g) * lf + g)
    
    
def toBody(state, vec):
    euler_angles = state[3], state[5], 0 # state[7]
    m = euler_to_rotmatrix(euler_angles[2], euler_angles[1], euler_angles[0])
    return m.T * Matrix(vec)
    
    
def aerodynamic_wrench(state):
    global C_mxy, C_mz, C_wxy, C_wz
    euler = state[3], state[5], 0 # state[7]
    eulerd = state[4], state[6], state[7]
    R = euler_to_rotmatrix(euler[2], euler[1], euler[0])
    AdR = np.block([[R, np.zeros([3, 3])], [np.zeros([3, 3]), R]])
    wb = eulerd2w(euler, eulerd)
    v = state[0], state[1], state[2]
    body_twist = list(wb) + list(R.T @ np.array(v))
    twist = AdR @ body_twist
    u = np.array([twist[0], -twist[1], -twist[2], twist[3], -twist[4], -twist[5]])
    ub = AdR.T @ u
    wb, vb = np.sqrt(np.sum(np.square(ub[:3]))), np.sqrt(np.sum(np.square(ub[3:])))
    drag = [C_mxy*ub[0]*wb, C_mxy*ub[1]*wb, C_mz*ub[2]*wb, C_wxy*ub[3]*vb, C_wxy*ub[4]*vb, C_wz*ub[5]*vb]
    wrench = [-drag[0], drag[1], drag[2], -drag[3], drag[4], drag[5]]
    #print("aerodynamic wrench: ", wrench)
    return wrench


def get_torques(acceleration_command, state, command, rel_force):
    global I, axy_kp, az_kp, axy_ki, az_ki, gamma_wxy, gamma_wz, g, r, trigger, dt
    acceleration_command_body = toBody(state, acceleration_command)                          # body frame
    print('acb: ', acceleration_command_body)
    uwx_body, uwy_body = -acceleration_command_body[1]/g, acceleration_command_body[0]/g
    prev_uwx_body, prev_uwy_body = state[8], state[9]
    diff_uwb = [(uwx_body - prev_uwx_body)/dt, (uwy_body - prev_uwy_body)/dt]
    command_uwz = command[-1]
    euler = state[3], state[5], 0 # state[7]
    eulerd = state[4], state[6], state[7]
    wx, wy, wz = eulerd2w(euler, eulerd)
    #error_wx, error_wy, error_wz = state[-3:]
    if trigger:
        #torque_x = I[0, 0] * (axy_kp * uwx_body + axy_ki*error_wx + axy_kd * diff_uwb[0])
        #torque_y = I[1, 1] * (axy_kp * uwy_body + axy_ki*error_wy + axy_kd * diff_uwb[1])
        torque_x = I1 * (axy_kp * uwx_body + axy_kd * diff_uwb[0])
        torque_y = I2 * (axy_kp * uwy_body + axy_kd * diff_uwb[1])
    else:
        #torque_x = I[0, 0] * (axy_kp * uwx_body + axy_ki*error_wx - axy_kd * wx)
        #torque_y = I[1, 1] * (axy_kp * uwy_body + axy_ki*error_wy - axy_kd * wy)
        torque_x = I1 * (axy_kp * uwx_body - axy_kd * wx)
        torque_y = I2 * (axy_kp * uwy_body - axy_kd * wy)
        diff_uwb = [-wx, -wy]
        trigger = True
    torque_z = I3 * az_kp * (command_uwz - wz) # + az_ki*error_wz
    torques = [torque_x, torque_y, torque_z]
    #d_error = [uwx_body, uwy_body, command_uwz - wz]
    return torques, diff_uwb # , d_error
    
    
def get_euler_accs(state, torques, aw0):
    global mass, I
    euler = state[3], state[5], 0 # state[7]
    eulerd = state[4], state[6], state[7]
    v = state[0], state[1], state[2]
    R = euler_to_rotmatrix(euler[2], euler[1], euler[0])
    wb = eulerd2w(euler, eulerd)
    body_twist = list(wb) + list(R.T * Matrix(v))
    
    for i in range(3):
        torques[i] += aw0[i] 
    
    d_body_twist_angular = []
    d_body_twist_angular.append((torques[0] + body_twist[1]*body_twist[2]*(I2 - I3))/I1)
    d_body_twist_angular.append((torques[1] + body_twist[2]*body_twist[0]*(I3 - I1))/I2)
    d_body_twist_angular.append((torques[2] + body_twist[0]*body_twist[1]*(I1 - I2))/I3)
    #d_body_twist.append(body_twist[2]*body_twist[4] - body_twist[1]*body_twist[5])
    #d_body_twist.append(body_twist[5]*body_twist[0] - body_twist[3]*body_twist[2])
    #d_body_twist.append(body_twist[3]*body_twist[1] - body_twist[4]*body_twist[0])
    
    eulerdd = wd2eulerdd(euler, eulerd, np.array(d_body_twist_angular))
    return eulerdd
    
    
def get_cartesian_accs(state, rel_force_command, aw1):
    global g, mass
    roll, pitch, yaw = state[3], state[5], 0 # state[7]
    rolld, pitchd, yawd = state[4], state[6], state[7]
    v = state[0], state[1], state[2]                  # world speeds
    R = euler_to_rotmatrix(yaw, pitch, roll)
    wb = eulerd2w([roll, pitch, yaw], [rolld, pitchd, yawd])
    body_twist = list(wb) + list(R.T @ np.array(v))
    wx, wy, wz, vx, vy, vz = body_twist
    thrust = np.array([aw1[0]/mass, aw1[1]/mass, (rel_force_command + aw1[2]) / mass])
    #print("NORM w: ", np.sqrt(np.sum(np.square([wx, wy, wz]))))
    a_b = thrust - np.cross([wx, wy, wz], [vx, vy, vz])
    accs = R @ a_b + np.array([0, 0, -g])
    return [accs[0], accs[1], accs[2]]
    
    
def local(state, output, diff_uwb):
    roll, pitch, yaw = state[3], state[5], 0 # state[7]
    rolld, pitchd, yawd = state[4], state[6], state[7]
    R = euler_to_rotmatrix(yaw, pitch, roll)
    dX = R.T * Matrix(output[:3])
    final_output = list(dX) + [rolld] + [output[3]] + [pitchd] + [output[4]] + [output[5]] + diff_uwb # + d_error_1 + d_error_2
    return final_output


def forward(raw_x, raw_u):
    x, u = toWorld(raw_x), stable_frame(raw_x, raw_u)
    acc_commands, d_error_xyz = get_acceleration_commands(x, u)
    acc_command_body = toBody(x, acc_commands)
    relative_z_force = get_force_z(x, acc_command_body[2])
    torques, diff_uwb = get_torques(acc_commands, x, u, relative_z_force)
    #aw = aerodynamic_wrench(x)
    aw = [0, 0, 0, 0, 0, 0]
    output_euler_acc = get_euler_accs(x, torques, aw[:3])
    output_cart_acc = get_cartesian_accs(x, relative_z_force, aw[3:])
    output_acc = list(output_cart_acc) + list(output_euler_acc)
    local_output_acc = local(x, output_acc, diff_uwb)
    return local_output_acc
    
    
    
    
    
if __name__ == "__main__":
    
    try:
        interval_mode = bool(int(sys.argv[1]))
    except:
        pass
    
    input_X = [d_xq, d_yq, d_z, roll, rolld, pitch, pitchd, yawd, prev_uwx_b, prev_uwy_b]
    input_U = [ux, uy, uz, uwz]
    output = forward(input_X, input_U)
    real_output = [output[0], output[1], output[2], output[4], output[6], output[7], output[8], output[9]]
    for i in range(10):
        print(f'OUTPUT[{i}]:\n', output[i])
    #outputs = Matrix(output)      # len 10
    values = {d2_xq: 0, d2_yq: 0, d2_z: 0, rolld: 0, rolldd: 0, pitchd: 0, pitchdd: 0, yawdd: 0, d_prev_uwx_b: 0, d_prev_uwy_b: 0}  # Equilibrium
    other_vars = {'g': 9.8065, 'lxy_kp': 5, 'lz_kp': 5, 'axy_kp': 10, 'az_kp': 5, 'axy_kd': 5}
    if not interval_mode:
        other_vars.update({'I1': I[0, 0], 'I2': I[1, 1], 'I3': I[2, 2], 'mass': _mass})
        der_doc = 'reduced_partial_derivatives_values'
        formula_doc = 'reduced_accelerations_formulas'
    else:
        der_doc = 'INTERVAL_reduced_partial_derivatives_values'
        formula_doc = 'INTERVAL_reduced_accelerations_formulas'
    values.update(other_vars)
    
    print('MODE: CONTINUOUS')
    file1 = f'/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/yaml/Linear_Coefficients/{der_doc}.yaml'
    file2 = f'/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/yaml/Linear_Coefficients/{formula_doc}.yaml'
    print('der doc', file1)
    print('formula doc', file2)
    _elts = input_X + input_U
    accelerations = [d2_xq, d2_yq, d2_z, rolldd, pitchdd, yawdd, d_prev_uwx_b, d_prev_uwy_b]  # len 10
        
        
    with open(file1, 'w') as f:
        data = dict()
        #print("VARS: ", _vars)
        #print("ELTS: ", _elts)
        #print("OUTS: ", _outs[-1])
        for i, v in enumerate(accelerations):
            sub_data = dict()
            for j, elt in enumerate(_elts):
                if elt not in [d2_xq, d2_yq, d2_z, rolldd, pitchdd, yawdd, d_prev_uwx_b, d_prev_uwy_b]:
                    der = Derivative(real_output[i], elt)
                    res = der.doit()
                    sub_data[str(elt)] = str(simplify(res.subs(values)))
                    print(v, ' ', elt)
            data[str(v)] = sub_data
        data['rolld'] = {str(key): 0 for key in _elts}
        data['pitchd'] = {str(key): 0 for key in _elts}
        data['rolld']['rolld'] = 1
        data['pitchd']['pitchd'] = 1
        print(data)
        yaml.dump(data, f, default_flow_style=None)
    with open(file2, 'w') as g:
        data = dict()
        for i, v in enumerate(accelerations):
            data[str(v)] = str(real_output[i])
        yaml.dump(data, g, default_flow_style=None)
        
        
        
    '''    
    with open(f'/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/yaml/Linear_Coefficients/{der_doc}.yaml', 'w') as f:
        data = dict()
        for i, v in enumerate(new_X):
            sub_data = dict()
            for j, elt in enumerate(new_elements):
                if j % 3 != 2:
                    if i % 3 == 2:
                        der = Derivative(new_outputs[i], elt)
                        res = der.doit()
                        sub_data[str(elt)] = str(simplify(res.subs(values)))
                        print(v, ' ', elt)
                    else:
                        sub_data[str(elt)] = 0
                        if i == j:
                            sub_data[str(elt)] = 1
                    data[str(v)] = sub_data
        yaml.dump(data, f, default_flow_style=None)
    
    new_accs = [d2_xq, d2_yq, d2_z, rolldd, pitchdd, yawdd]
    with open(f'/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/yaml/Linear_Coefficients/{formula_doc}.yaml', 'w') as g:
        data = dict()
        for i, v in enumerate(new_accs):
            data[str(v)] = str(outputs[3*i+2])
        yaml.dump(data, g, default_flow_style=None)'''
