import numpy as np
from math import atan2
import matplotlib.pyplot as plt
from casadi import *
import yaml
import sys
from quadrotor_mpc.transform_utils import euler_to_quaternion, VecToso3, euler_to_rotmatrix, eulerd2w, wd2eulerdd, cross, quaternion_to_rotmatrix



g = 9.8065
dt = 0.01

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
#gamma_xy, gamma_z, gamma_wx, gamma_wy, gamma_wz = 5.1, 5.1, 10.5, 10.5, 5.25
lxy_kp, lz_kp, axy_kp, az_kp = 5, 5, 10, 5
lxy_ki, lz_ki, axy_ki, az_ki = 1, 1, 5, 2.5
#lxy_ki, lz_ki, axy_ki, az_ki = 0, 0, 0, 0
axy_kd = 5
#r = [2/3, 2/3, 2/3, 10/11, 10/11, 0.5]
#r = [1/6, 1/6, 1/6, 1/2, 1/2, 1/11]
use_Interval = False

d_xq, d_yq, d_z, roll, rolld, pitch, pitchd, yawd = MX.sym('d_xq'), MX.sym('d_yq'), MX.sym('d_z'), MX.sym('roll'), MX.sym('rolld'), MX.sym('pitch'), MX.sym('pitchd'), MX.sym('yawd')
prev_uwx_b, prev_uwy_b = MX.sym('prev_uwx_b'), MX.sym('prev_uwy_b')
                                                                                                                 
ux, uy, uz, uwz = MX.sym('ux'), MX.sym('uy'), MX.sym('uz'), MX.sym('uwz')

trigger = False

C_mxy = 0.0741562
C_mz = 0.0506433
C_wxy = 0.12
C_wz = 0.1

#prev_uwx_body, prev_uwy_body = 0, 0

n = 3
_I1, _I2, _I3 = I1, I2, I3
margin = 0.003




def n2tab(i, n, q):
    k = 1
    t = np.zeros(q)
    while True:
        t[k-1] = i // (n**(q-k))
        i = i % (n**(q-k))
        k += 1
        if k > q:
            break
    return t


def val(tab, i):
    global n, _I1, _I2, _I3, margin
    T = n2tab(i, n, 3)
    I1, I2, I3 = _I1 + (T[0]-1)*margin, _I1 + (T[1]-1)*margin, _I1 + (T[2]-1)*margin
    return I1, I2, I3






def stable_frame(x, u):
    yaw = x[7]   #state[7] useless
    real_u = [0, 0, 0, 0]
    real_u[0] = np.cos(yaw) * u[0] - np.sin(yaw) * u[1]
    real_u[1] = np.sin(yaw) * u[0] + np.cos(yaw) * u[1]
    real_u[2] = u[2]
    real_u[3] = u[3]
    return real_u


def toWorld(x):
    new_x = [None for _ in range(17)]
    # positions are useless
    roll, pitch, yaw = x[3], x[5], x[7]
    R = euler_to_rotmatrix(yaw, pitch, roll)
    X = R @ x[:3]
    new_x[:3] = X
    #new_x[0] = x[0]*np.cos(yaw) - x[1]*np.sin(yaw)
    #new_x[1] = x[0]*np.sin(yaw) + x[1]*np.cos(yaw)
    for i in range(2, len(new_x)):
        new_x[i] = x[i]
    return new_x
    
    
def get_acceleration_commands(state, command):
    global lxy_kp, lz_kp, lxy_ki, lz_ki, gamma_xy, gamma_z, r, g, dt
    v = state[:3]
    error_x, error_y, error_z = state[11:14]
    command_ux = command[0]
    acceleration_command_x = lxy_kp * (command_ux - v[0]) #+ lxy_ki * error_x
    command_uy = command[1]
    acceleration_command_y = lxy_kp * (command_uy - v[1]) #+ lxy_ki * error_y
    command_uz = command[2]
    acceleration_command_z = lz_kp * (command_uz - v[2]) + g #+ lz_ki * error_z
    acceleration_command = [acceleration_command_x, acceleration_command_y, acceleration_command_z]
    d_error = [command_ux - v[0], command_uy - v[1], command_uz - v[2]]
    return acceleration_command, d_error
    
    
def load_factor(roll, pitch):
    return 1 / (np.cos(roll)*np.cos(pitch))
    
    
def get_force_z(state, acbz):
    global g, gamma_z, r
    lf = load_factor(state[3], state[5])
    return mass * ((acbz - g) * lf + g)
    
    
def toBody(state, vec):
    euler_angles = state[3], state[5], state[7]
    m = euler_to_rotmatrix(euler_angles[2], euler_angles[1], euler_angles[0])
    return np.dot(m.T, vec)
    
    
def aerodynamic_wrench(state):
    global C_mxy, C_mz, C_wxy, C_wz
    euler = state[3], state[5], state[7]
    eulerd = state[4], state[6], state[8]
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
    uwx_body, uwy_body = -acceleration_command_body[1]/g, acceleration_command_body[0]/g
    prev_uwx_body, prev_uwy_body = state[9], state[10]
    diff_uwb = [(uwx_body - prev_uwx_body)/dt, (uwy_body - prev_uwy_body)/dt]
    command_uwz = command[-1]
    euler = state[3], state[5], state[7]
    eulerd = state[4], state[6], state[8]
    wx, wy, wz = eulerd2w(euler, eulerd)
    error_wx, error_wy, error_wz = state[-3:]
    if trigger:
        #torque_x = I1 * (axy_kp * uwx_body + axy_ki*error_wx + axy_kd * diff_uwb[0])      # DELAY APPROXIMATION
        #torque_y = I2 * (axy_kp * uwy_body + axy_ki*error_wy + axy_kd * diff_uwb[1])
        torque_x = I1 * (axy_kp * uwx_body + axy_kd * diff_uwb[0])      # DELAY APPROXIMATION
        torque_y = I2 * (axy_kp * uwy_body + axy_kd * diff_uwb[1])
        #torque_x = I[0, 0] * (axy_kp * 2*np.arctan(uwx_body) + axy_kd * 2*np.arctan(diff_uwb[0]))      # DELAY APPROXIMATION
        #torque_y = I[1, 1] * (axy_kp * 2*np.arctan(uwy_body) + axy_kd * 2*np.arctan(diff_uwb[1]))
    else:
        #torque_x = I1 * (axy_kp * uwx_body + axy_ki*error_wx - axy_kd * wx)
        #torque_y = I2 * (axy_kp * uwy_body + axy_ki*error_wy - axy_kd * wy)
        torque_x = I1 * (axy_kp * uwx_body - axy_kd * wx)
        torque_y = I2 * (axy_kp * uwy_body - axy_kd * wy)
        #torque_x = I[0, 0] * (axy_kp * 2*np.arctan(uwx_body) - axy_kd * 2*np.arctan(wx))
        #torque_y = I[1, 1] * (axy_kp * 2*np.arctan(uwy_body) - axy_kd * 2*np.arctan(wy))
        diff_uwb = [-wx, -wy]
        trigger = True
    torque_z = I3 * az_kp * (command_uwz - wz) + az_ki*error_wz
    torques = [torque_x, torque_y, torque_z]
    d_error = [uwx_body, uwy_body, command_uwz - wz]
    return torques, diff_uwb, d_error
    
    
def get_euler_accs(state, torques, aw0):
    global mass, I
    euler = state[3], state[5], state[7]
    eulerd = state[4], state[6], state[8]
    v = state[0], state[1], state[2]
    R = euler_to_rotmatrix(euler[2], euler[1], euler[0])
    wb = eulerd2w(euler, eulerd)
    body_twist = list(wb) + list(R.T @ np.array(v))
    
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
    roll, pitch, yaw = state[3], state[5], state[7]
    rolld, pitchd, yawd = state[4], state[6], state[8]
    v = state[0], state[1], state[2]                  # world speeds
    R = euler_to_rotmatrix(yaw, pitch, roll)
    wb = eulerd2w([roll, pitch, yaw], [rolld, pitchd, yawd])
    body_twist = list(wb) + list(R.T @ np.array(v))
    wx, wy, wz, vx, vy, vz = body_twist
    #w = np.sqrt(np.sum(np.square([wx, wy, wz])))
    #angle = np.pi/2
    #if w != 0:
    #    q = [np.cos(angle/2), wx*np.sin(angle/2)/w, wy*np.sin(angle/2)/w, wz*np.sin(angle/2)/w]
    #    vec = np.array([vx, vy, vz]) @ quaternion_to_rotmatrix(q)
    #else:
    #    vec = [0, 0, 0]
    #print(vec, w)
    #accs = np.array([rel_force_command/mass * (np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)) + vec[0]*w,
    #                 rel_force_command/mass * (np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)) + vec[1]*w,
    #                 rel_force_command/mass * np.cos(pitch)*np.cos(roll) - g + vec[2]*w])
    thrust = np.array([aw1[0]/mass, aw1[1]/mass, (rel_force_command + aw1[2]) / mass])
    #print("NORM w: ", np.sqrt(np.sum(np.square([wx, wy, wz]))))
    a_b = thrust - np.cross([wx, wy, wz], [vx, vy, vz])
    accs = R @ a_b + np.array([0, 0, -g])
    return [accs[0], accs[1], accs[2]]
    
    
def local(state, output, diff_uwb, d_error_1, d_error_2):
    roll, pitch, yaw = state[3], state[5], state[7]
    rolld, pitchd, yawd = state[4], state[6], state[8]
    R = euler_to_rotmatrix(yaw, pitch, roll)
    dX = R.T @ output[:3]
    final_output = dX.tolist() + [rolld] + [output[3]] + [pitchd] + [output[4]] + [yawd] + [output[5]] + diff_uwb + d_error_1 + d_error_2
    return final_output


def forward(raw_x, raw_u):
    x, u = toWorld(raw_x), stable_frame(raw_x, raw_u)
    acc_commands, d_error_xyz = get_acceleration_commands(x, u)
    acc_command_body = toBody(x, acc_commands)
    relative_z_force = get_force_z(x, acc_command_body[2])
    torques, diff_uwb, d_error_w = get_torques(acc_commands, x, u, relative_z_force)
    #aw = aerodynamic_wrench(x)
    aw = [0, 0, 0, 0, 0, 0]
    output_euler_acc = get_euler_accs(x, torques, aw[:3])
    output_cart_acc = get_cartesian_accs(x, relative_z_force, aw[3:])
    output_acc = list(output_cart_acc) + list(output_euler_acc)
    local_output_acc = local(x, output_acc, diff_uwb, d_error_xyz, d_error_w)
    return local_output_acc
    
    
if __name__ == "__main__":

    #input_x = vertcat(d_xq, d_yq, d_z, roll, rolld, pitch, pitchd, yawd, prev_uwx_b, prev_uwy_b)
    #input_u = vertcat(ux, uy, uz, uwz)
    #if bool(int(sys.argv[1])):
    #    I1, I2, I3, mass = MX.sym('I00'), MX.sym('I11'), MX.sym('I22'), MX.sym('mass')
    #    input_y = vertcat(I1, I2, I3, mass)
    #    print(input_x)
    #    print(input_y)
    #    output = forward(input_x, input_u)
    #    f = Function('f', [input_x, input_u, input_y], output)
    #    f.save('CasADi_formulas/reduced_dyn_f_INTERVAL.casadi')
    #else:
    #    print(input_x)
    #    output = forward(input_x, input_u)
    #    #print(output)
    #    f = Function('f', [input_x, input_u], output)
    #    f.save('CasADi_formulas/reduced_dyn_f.casadi')

    
    try:
        test = int(sys.argv[1])
    except:
        test = 0
        
    print("test: ", test)
    trigger = False
    
    nx = 17
    
    if test == 0:
        index = [5]
        tab = []
        tab2 = []
        #dt = 0.001
        s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([1, 0, 0, 0])
        #d_s = np.array(forward(s, u))
        #s += d_s*dt
        #t = []
        #for ind in index:
        #    t.append(s[ind])
        #tab.append(t)
        trigger = True
        for i in range(2000):
            d_s = np.array(forward(s, u))
            #print(f'{i}: {d_s}')
            #offset = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, s[-2], s[-1]])
            s += d_s*dt #+ offset
            #print("STATE: ", s)
            t = []
            #R = euler_to_rotmatrix(s[7], s[5], s[3])
            #w = eulerd2w([s[3], s[5], s[7]], [s[4], s[6], s[8]])
            #sol = R @ w
            for ind in index:
                t.append(s[ind])
                #t.append(sol)
            tab.append(t)
            #tab.append(acc[0])
            #print(t)
        
        tab = np.array(tab)
        for i in range(len(index)):
            plt.plot(tab[:, i])
        #plt.plot(tab2)
        #plt.plot(tab[:, 1])
        #plt.plot(tab[:, 2])
        plt.show()
        
    elif test == 1:
        N = 1000
        trigger = False
        tab = [[] for _ in range(n**3)]
        full_tab = [[] for _ in range(n**3)]
        index = 2
        for j in range(n**3):
            I1, I2, I3 = val([_I1, _I2, _I3], j)
            print(I1, I2, I3)
            y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            u = np.array([2, 2, 2, 1])
            trigger = True
            for i in range(N):
                d_y = np.array(forward(y, u))
                y += d_y*dt #+ offset
                tab[j].append(y[index])
                full_tab[j].append(y.tolist())
        tab = np.array(tab)
        full_tab = np.array(full_tab)
        
        for j in range(n**3):
            plt.plot(tab[j, :])
        
        plt.show()
        
    else:
        N = 50
        trigger = False
        tab = [[] for _ in range(n**3)]
        full_tab = [[] for _ in range(n**3)]
        index = 2
        for j in range(n**3):
            I1, I2, I3 = val([_I1, _I2, _I3], j)
            print(I1, I2, I3)
            y = np.array([0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            u = np.array([1, 1, 1, 1])
            trigger = True
            for i in range(N):
                d_y = np.array(forward(y, u))
                y += d_y*dt #+ offset
                tab[j].append(y[index])
                full_tab[j].append(y.tolist())
        tab = np.array(tab)
        full_tab = np.array(full_tab)
        
        for j in range(n**3):
            for i in range(1, N):
                s = [np.random.choice(full_tab[:, i-1, r]) for r in range(nx)]
                #s = full_tab[j, i-1, :].tolist()
                k = np.random.randint(n**3)
                I1, I2, I3 = val([_I1, _I2, _I3], k)
                full_tab[j, i, :] = full_tab[j, i-1, :] + dt*np.array(forward(s, u))
            plt.plot(full_tab[j, :, index])
            
        plt.show()
