import numpy as np
from math import atan2
import matplotlib.pyplot as plt


nx = 20
nu = 10

OI = np.array([[1.15202001e-02,  0.00000000e+00, -1.27699200e-09],
               [0.00000000e+00,  1.15457173e-02,  0.00000000e+00],
               [-1.27699200e-09,  0.00000000e+00,  2.18256172e-02]])
I = np.array([[0.0115202, 0, 0],
              [0, 0.0115457, 0],
              [0, 0, 0.0218256]])
g = 9.8065
mass = 1.478
CoG = [-0.000108, 0, -8e-6]


period = 0.1
#dt = 0.01
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
r = [period / (period + lxy_tau), period / (period + lxy_tau), period / (period + lz_tau), period / (period + axy_tau), period / (period + axy_tau), period / (period + az_tau)]
print("gamma: ", gamma_x, " ", gamma_y, " ", gamma_z, " ", gamma_wx, " ", gamma_wy, " ", gamma_wz)
print("r: ", r)



def Rot(state, control):
    yaw = state[15]
    u1, u2 = control[1], control[3]
    control[1] = np.cos(yaw) * u1 - np.sin(yaw) * u2
    control[3] = np.sin(yaw) * u1 + np.cos(yaw) * u2
    return control


def VecToso3(omg):
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])
                     
                     
def eulerd2w(euler, eulerd):
    roll, pitch, yaw = euler
    m = np.array([[1,             0,             -np.sin(pitch)], 
                  [0,  np.cos(roll), np.cos(pitch)*np.sin(roll)], 
                  [0, -np.sin(roll), np.cos(pitch)*np.cos(roll)]])
    return m @ np.array(eulerd)
    
    
def wd2eulerdd(euler, eulerd, wd):
    roll, pitch, yaw = euler
    rolld, pitchd, yawd = eulerd
    minv = np.array([[1, np.tan(pitch)*np.sin(roll), np.tan(pitch)*np.cos(roll)], 
                     [0,               np.cos(roll),              -np.sin(roll)], 
                     [0, np.sin(roll)/np.cos(pitch), np.cos(roll)/np.cos(pitch)]])
    md = np.array([[0,                   0,                                                 -pitchd*np.cos(pitch)], 
                   [0, -rolld*np.sin(roll),  rolld*np.cos(pitch)*np.cos(roll) - pitchd*np.sin(roll)*np.sin(pitch)], 
                   [0, -rolld*np.cos(roll), -rolld*np.cos(pitch)*np.sin(roll) - pitchd*np.cos(roll)*np.sin(pitch)]])
    return minv @ (wd - md @ eulerd)
  
                     
def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qw, qx, qy, qz]
    
    
def euler_to_rotmatrix(yaw, pitch, roll):
    return np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.cos(roll)*np.sin(pitch) + np.sin(yaw)*np.sin(roll)], 
                     [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.cos(roll)*np.sin(pitch) - np.cos(yaw)*np.sin(roll)], 
                     [            -np.sin(pitch),                                       np.sin(roll)*np.cos(pitch),                                        np.cos(roll)*np.cos(pitch)]])



def load_factor(euler_angles):
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
    q = euler_to_quaternion(yaw, pitch, roll);
    return 1 / (q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)
    
    
def twist_body_angular(state):
    roll, pitch, yaw = state[9], state[12], state[15]
    rolld, pitchd, yawd = state[10], state[13], state[16]
    wb = eulerd2w([roll, pitch, yaw], [rolld, pitchd, yawd])
    return wb


def toBody(state, vec):
    roll, pitch, yaw = state[9], state[12], state[15]
    m = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.cos(roll)*np.sin(pitch) + np.sin(yaw)*np.sin(roll)], 
                  [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.cos(roll)*np.sin(pitch) - np.cos(yaw)*np.sin(roll)], 
                  [           -np.sin(pitch),                                        np.sin(roll)*np.cos(pitch),                                        np.cos(roll)*np.cos(pitch)]])
    vec_body = m.T @ vec
    return vec_body
    
    
def get_acceleration_commands(state, command):
    global gamma_x, gamma_y, gamma_z, r, g
    cart_speeds = [state[1], state[4], state[7]]
    if command[0] is not None:
        acceleration_command_x = -gamma_x * cart_speeds[0] + (1-r[0])*gamma_x*command[0] + r[0]*gamma_x*command[1]
    else:
        acceleration_command_x = -gamma_x * cart_speeds[0] + gamma_x*command[1]
    if command[2] is not None:
        acceleration_command_y = -gamma_y * cart_speeds[1] + (1-r[1])*gamma_y*command[2] + r[1]*gamma_y*command[3]
    else:
        acceleration_command_y = -gamma_y * cart_speeds[1] + gamma_y*command[3]
    if command[4] is not None:
        acceleration_command_z = -gamma_z * cart_speeds[2] + (1-r[2])*gamma_z*command[4] + r[2]*gamma_z*command[5] + g
    else:
        acceleration_command_z = -gamma_z * cart_speeds[2] + gamma_z*command[5] + g
    return np.array([acceleration_command_x, acceleration_command_y, acceleration_command_z])
    
    
def get_torques(acceleration_command, state, command, rel_force, tba):
    global I, gamma_wx, gamma_wy, gamma_wz, period, prev_tbx, prev_tby, g, r
    acceleration_command_body = toBody(state, acceleration_command)
    print("acceleration command body: ", acceleration_command_body)
    uwx_body_pass, uwy_body_pass = command[8], command[9]
    uwx_body, uwy_body = -acceleration_command_body[1]/g, acceleration_command_body[0]/g
    tbx, tby = tba[:2]
    prev_tbx, prev_tby = state[nx-2:]
    tbx, tby, prev_tbx, prev_tby = 0, 0, 0, 0
    uwz_pass, uwz = command[6], command[7]
    wz = eulerd2w([state[9], state[12], state[15]], [state[10], state[13], state[16]])[2]
    if prev_tbx is not None and uwx_body_pass is not None:
        lowpass_uwx_body = (1-r[3]) * uwx_body_pass + r[3] * uwx_body
        torque_x = I[0, 0] * (gamma_wx * lowpass_uwx_body + axy_kd * ((lowpass_uwx_body - uwx_body_pass)/period - (tbx - prev_tbx)))
        print("low pass uwx body, current and pass: ", lowpass_uwx_body, uwx_body_pass)
        print("tbx, current and pass: ", tbx, prev_tbx)
        print("torque x: ", torque_x)
    else:
        torque_x = I[0, 0] * (gamma_wx * uwx_body - axy_kd * tbx)
    if prev_tby is not None and uwy_body_pass is not None:
        lowpass_uwy_body = (1-r[4]) * uwy_body_pass + r[4] * uwy_body
        torque_y = I[1, 1] * (gamma_wy * lowpass_uwy_body + axy_kd * ((lowpass_uwy_body - uwy_body_pass)/period - (tby - prev_tby)))
        print("low pass uwy body, current and pass: ", lowpass_uwy_body, uwy_body_pass)
        print("tby, current and pass: ", tby, prev_tby)
        print("torque y: ", torque_y)
    else:
        torque_y = I[1, 1] * (gamma_wy * uwy_body - axy_kd * tby)
    if uwz_pass is not None:
        torque_z = I[2, 2] * gamma_wz * (-wz + (1-r[5]) * uwz_pass + r[5] * uwz)
    else:
        torque_z = I[2, 2] * gamma_wz * (-wz + uwz)
    torques = np.array([torque_x, torque_y, torque_z])
    force_offset = np.zeros(3)
    force_offset[0] = CoG[1]*rel_force
    force_offset[1] = -CoG[0]*rel_force
    torques[0] -= force_offset[0]
    torques[1] -= force_offset[1]
    return torques
    
    
def get_force(state, acceleration_command_z):
    global g, mass
    euler_angles = np.array([state[9], state[12], state[15]])
    lf = load_factor(euler_angles)
    print("load factor: ", lf)
    return mass * ((acceleration_command_z - g) * lf + g);


def Ad_Twist(twist):
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
    Ad[3:, :3] = VecToso3(pos) @ R
    Ad[3:, 3:] = R
    return Ad
    
def AdjointInvert(state):
    pos = state[:9:3]
    roll, pitch, yaw = state[9], state[12], state[15] 
    AdI = np.zeros([6, 6])
    R = euler_to_rotmatrix(yaw, pitch, roll)
    AdI[:3, :3] = R.T
    AdI[:3, 3:] = np.zeros([3, 3])
    AdI[3:, :3] = -R.T @ VecToso3(pos)
    AdI[3:, 3:] = R.T
    return AdI
    
    
def get_euler_accs(state, torques):
    global mass, CoG, OI
    euler = state[9:16:3]
    eulerd = state[10:17:3]
    speeds = np.zeros(3)
    wrench_body = np.zeros(6)
    for i in range(3):    
        speeds[i] = state[3*i+1]
        wrench_body[i] = torques[i]
    wb = eulerd2w(euler, eulerd).tolist()
    vb = list(R.T @ np.array(speeds))
    body_twist = np.array(wb + vb)
    G = np.zeros([6, 6])
    G[:3, :3] = OI
    G[:3, 3:] = mass * VecToso3(CoG)
    G[3:, :3] = -mass * VecToso3(CoG)
    G[3:, 3:] = mass * np.eye(3)
    der_body_twist = np.linalg.inv(G) @ (wrench_body + Ad_Twist(body_twist).T @ G @ body_twist)
    d_wb = der_body_twist[:3]
    eulerdd = wd2eulerdd(euler, eulerd, d_wb)
    print("ang_acc: ", eulerdd)
    return eulerdd
    
    
def get_cartesian_accs(state, rel_force_command):
    global mass, g
    roll, pitch, yaw = state[9], state[12], state[15]
    acc_x = rel_force_command/mass * (np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll))
    acc_y = rel_force_command/mass * (np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll))
    acc_z = rel_force_command/mass * np.cos(pitch)*np.cos(roll) - g
    return np.array([acc_x, acc_y, acc_z])
    
    
def state_vec_from_acc(state, acc_vec, twist_body):
    state_vec = np.zeros(nx)
    for i in range(6):
        state_vec[3*i]   = state[3*i] + period * state[3*i+1] + (period**2)/2 * acc_vec[i];
        state_vec[3*i+1] = state[3*i+1] + period * acc_vec[i];
        state_vec[3*i+2] = acc_vec[i];
    state_vec[nx-2] = twist_body[0]
    state_vec[nx-1] = twist_body[1]
    return state_vec
    
    
def forward(x, u):
    acc_commands = get_acceleration_commands(x, u)
    print("acceleration commands: ", acc_commands)
    tba = twist_body_angular(x)
    print("twist body: ", tba)
    relative_z_force = get_force(x, acc_commands[2])
    print("force: ", relative_z_force)
    planar_torques = get_torques(acc_commands, x, u, relative_z_force, tba)
    print("torques: ", planar_torques)
    output_cart_acc = get_cartesian_accs(x, relative_z_force)
    output_euler_acc = get_euler_accs(x, planar_torques)
    output_acc = np.concatenate([output_cart_acc, output_euler_acc])
    print("output_accelerations: ", output_acc)
    output = state_vec_from_acc(x, output_acc, tba)
    return output



def test():
    global g
    x0 = np.zeros(20)
    x0[6] = 0.18
    x0[-2], x0[-1] = None, None
    ugross = np.array([None, 0, None, 0, None, 1, None, 0, None, None])
    u0 = Rot(x0, ugross)
    output = forward(x0, u0)
    print(output)
    while True:
        ac_command = get_acceleration_commands(x0, u0)
        ac_command_body = toBody(x0, ac_command)
        if u0[-2] is not None and u0[-1] is not None:
            prev_x, prev_y = u0[-2:]
            u0[-2:] = -ac_command_body[1]/g * r[3] + prev_x * (1-r[3]), ac_command_body[0]/g * r[4] + prev_y * (1-r[4])
        else:
            u0[-2:] = -ac_command_body[1]/g, ac_command_body[0]/g
        if None not in u0[:6]:
            u0[:6:2] = [r[i]*u0[2*i+1] + (1-r[i])*u0[2*i] for i in range(3)]
        else:
            for i in range(3):
                u0[2*i] = u0[2*i+1]
        if u0[6] is not None:
            u0[6] = r[5]*u0[7] + (1-r[5])*u0[6]
        else:
            u0[6] = u0[7]
        print('\n')
        try:
            u0[1:-2:2] = [float(input(f'control {i}: ')) for i in range(4)]
        except:
            break
        print("Designed control: ", u0)
        u0 = Rot(x0, u0)
        x0 = output
        output = forward(x0, u0)
        print(output)
        

def iteration(x, u, output, new_u):
    global g
    ac_command = get_acceleration_commands(x, u)
    ac_command_body = toBody(x, ac_command)
    if u[-2] is not None and u[-1] is not None:
        prev_x, prev_y = u[-2:]
        u[-2:] = -ac_command_body[1]/g * r[3] + prev_x * (1-r[3]), ac_command_body[0]/g * r[4] + prev_y * (1-r[4])
    else:
        u[-2:] = -ac_command_body[1]/g, ac_command_body[0]/g
    if None not in u[:6]:
        u[:6:2] = [r[i]*u[2*i+1] + (1-r[i])*u[2*i] for i in range(3)]
    else:
        for i in range(3):
            u[2*i] = u[2*i+1]
    if u[6] is not None:
        u[6] = r[5]*u[7] + (1-r[5])*u[6]
    else:
        u[6] = u[7]
    u[1:-2:2] = new_u
    print("Designed control: ", u)
    u = Rot(x, u)
    x = output
    output = forward(x, u)
    print(output)
    return x, u, output


def stream(return_param=0):
    arr = []
    x0 = np.zeros(20)
    x0[6] = 0.18
    x0[-2], x0[-1] = None, None
    ugross = np.array([None, 0, None, 0, None, 1, None, 0, None, None])
    u0 = Rot(x0, ugross)
    output = forward(x0, u0)
    x, u = output, u0
    arr.append(output[return_param])
    for _ in range(200):
        x, u, output = iteration(x, u, output, [0, 0, 1, 0])
        arr.append(output[return_param])
    for _ in range(200):
        x, u, output = iteration(x, u, output, [0, 0, 0, 0])
        arr.append(output[return_param])
    for _ in range(200):
        x, u, output = iteration(x, u, output, [1, 0, 0, 0])
        arr.append(output[return_param])
    return np.array(arr)
        
if __name__ == "__main__":
    #tab = stream(0)
    #plt.pyplot(tab, 0.01*np.arange(len(tab)))
    #plt.show()
    test()
