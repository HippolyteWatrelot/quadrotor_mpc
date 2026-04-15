#! /usr/bin/env python3

import numpy as np
from math import atan2


def RotInv(R):
    return np.array(R).T


def TransInv(T):
    R, p = TransToRp(T)
    Rt = RotInv(R)
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]


def MatrixLog3(R):
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)


def MatrixLog6(T):
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                           [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                           np.dot(np.eye(3) - omgmat / 2.0 \
                           + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                              * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                               T[1][3],
                                                               T[2][3]])],
                     [[0, 0, 0, 0]]]


def MatrixExp3(so3mat):
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)


def MatrixExp6(se3mat):
    se3mat = np.array(se3mat)
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    if NearZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[MatrixExp3(se3mat[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta \
                                  + (1 - np.cos(theta)) * omgmat \
                                  + (theta - np.sin(theta)) \
                                    * np.dot(omgmat,omgmat),
                                  se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]


def JacobianBody(Blist, thetalist):
    Jb = np.array(Blist).copy().astype(float)
    T = np.eye(4)
    for i in range(len(thetalist) - 2, -1, -1):
        T = np.dot(T,MatrixExp6(VecTose3(np.array(Blist)[:, i + 1] \
                                         * -thetalist[i + 1])))
        Jb[:, i] = np.dot(Adjoint(T), np.array(Blist)[:, i])
    return Jb


def FKinBody(M, Blist, thetalist):
    T = np.array(M)
    for i in range(len(thetalist)):
        T = np.dot(T, MatrixExp6(VecTose3(np.array(Blist)[:, i] \
                                          * thetalist[i])))
    return T


def Adjoint(T):
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(VecToso3(p), R), R]]


def AxisAng3(expc3):
    return (Normalize(expc3), np.linalg.norm(expc3))


def Normalize(V):
    return V / np.linalg.norm(V)


def NearZero(z):
    return abs(z) < 1e-6


def TransToRp(T):
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]


def so3ToVec(so3mat):
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])


def VecToso3(omg):
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])


def VecTose3(V):
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.zeros((1, 4))]


def se3ToVec(se3mat):
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                 [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]





def right_angles(thetalist):
    for j in range(len(thetalist)):
        thetalist[j] = (thetalist[j] + np.pi) % (2 * np.pi) - np.pi
    return thetalist

def joints_limits_satisfied(current_joints, previous_joints, dt):
    global Joints_limits
    for i in range(len(Joints_Limits)):
        if current_joints[i] < Joints_Limits[i, 0] or current_joints[i] > Joints_Limits[i, 1] or np.abs((current_joints[i] - previous_joints[i]) % np.pi) / dt > Joints_Limits[i, 2]:
            return False
    return True

def reshape_to_transmatrix(vec):
    mat = np.zeros([4, 4])
    mat[-1, -1] = 1
    mat[:3, :3] = np.reshape(vec[:9], (3, 3))
    mat[:3, 3] = vec[9:]
    #pre_mat = FKinBody(mat, Blist, thetalist)
    return mat

def NextState(config, var_vec, dt=0.01, max_ang_speed=100):
    new_config = config
    for i in range(len(var_vec)):
        if abs(var_vec[i]) > max_ang_speed:
            print("ERROR : too large joint speed detected !")
            return 0
        new_config[i] += dt * var_vec[i]
    return new_config

def assert_joints_positions_rules(thetalist, epsilon=0.01):
    indices = []
    test = True
    for i in range(len(Joints_Limits)):
        if thetalist[i] < Joints_Limits[i, 0] + epsilon or thetalist[i] > Joints_Limits[i, 1] - epsilon:
            indices.append(i)
            test = False
    return test, indices
    
def quaternion_to_rotmatrix(q):
    """"w x y z"""
    assert len(q) == 4
    w, x, y, z = q
    return np.array([[2*(w**2 + x**2) - 1, 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 2*(w**2 + y**2) - 1, 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 2*(w**2 + z**2) - 1]])
    
                     
def sign(x):
    return x / np.abs(x)

def pose_to_transmatrix(p, q):
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotmatrix(q)
    T[:3, 3] = p
    return T

def posestamped_to_transmatrix(ps):
    T = np.eye(4)
    position = ps.pose.position
    orientation = ps.pose.orientation
    p = [position.x, position.y, position.z]
    q = [orientation.w, orientation.x, orientation.y, orientation.z]
    T[:3, :3] = quaternion_to_rotmatrix(q)
    T[:3, 3] = p
    return T

def transmatrix_to_pose(T):
    trace = T[0, 0] + T[1, 1] + T[2, 2]
    if trace > 0:
        S = np.sqrt(1 + trace) * 2
        qw = S / 4
        qx = (T[2, 1] - T[1, 2]) / S
        qy = (T[0, 2] - T[2, 0]) / S
        qz = (T[1, 0] - T[0, 1]) / S
    elif (T[0, 0] > T[1, 1]) and (T[0, 0] > T[2, 2]):
        S = np.sqrt(1 + T[0, 0] - T[1, 1] - T[2, 2]) * 2
        qw = (T[2, 1] - T[1, 2]) / S
        qx = S / 4
        qy = (T[0, 1] + T[1, 0]) / S
        qz = (T[0, 2] + T[2, 0]) / S
    elif T[1, 1] > T[2, 2]:
        S = np.sqrt(1 + T[1, 1] - T[0, 0] - T[2, 2]) * 2
        qw = (T[0, 2] - T[2, 0]) / S
        qx = (T[0, 1] + T[1, 0]) / S
        qy = S / 4
        qz = (T[1, 2] + T[2, 1]) / S
    else:
        S = np.sqrt(1 + T[2, 2] - T[0, 0] - T[1, 1]) * 2
        qw = (T[1, 0] - T[0, 1]) / S
        qx = (T[0, 2] + T[2, 0]) / S
        qy = (T[1, 2] + T[2, 1]) / S
        qz = S / 4
    p = [T[0, 3], T[1, 3], T[2, 3]]
    q = [qw, qx, qy, qz]
    return p, q


def rotmatrix_to_quaternion(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        S = np.sqrt(1 + trace) * 2
        qw = S / 4
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = S / 4
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = S / 4
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = S / 4
    q = [qw, qx, qy, qz]
    return q
    
def xor(b1: bool, b2: bool):
    return not(b1 and b2) and b1 != b2
    
def rotmat_from_yaw(angle):
    a, b = np.cos(angle), np.sin(angle)
    return([[a, -b, 0], 
            [b,  a, 0], 
            [0,  0, 1]])
            
def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qw, qx, qy, qz]
    
def quaternion_to_euler(q):
    roll = atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    pitch = -np.pi/2 + 2*atan2(np.sqrt(1 + 2*(q[0]*q[2] - q[1]*q[3])), np.sqrt(1 - 2*(q[0]*q[2] - q[1]*q[3])))
    yaw = atan2(2*(q[0]*q[3] - q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
    return [roll, pitch, yaw]

def euler_to_rotmatrix(yaw, pitch, roll):
    return np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.cos(roll)*np.sin(pitch) + np.sin(yaw)*np.sin(roll)], 
                     [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.cos(roll)*np.sin(pitch) - np.cos(yaw)*np.sin(roll)], 
                     [            -np.sin(pitch),                                        np.sin(roll)*np.cos(pitch),                                        np.cos(roll)*np.cos(pitch)]])
                     
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
    
#def wd2eulerdd(euler, eulerd, wd):
#    roll, pitch, yaw = euler
#    rolld, pitchd, yawd = eulerd
#    w = eulerd2w(euler, eulerd)
#    rolldd = wd[0] + wd[1]*np.sin(roll)*np.tan(pitch) + wd[2]*np.cos(roll)*np.tan(pitch) + w[1]*(rolld*np.cos(roll)*np.tan(pitch) + pitchd*np.sin(roll)/(np.cos(pitch)**2)) + w[2]*(-rolld*np.sin(roll)*np.tan(pitch) + pitchd*np.cos(roll)/(np.cos(pitch)**2))
#    pitchdd = wd[1]*np.cos(roll) - wd[2]*np.sin(roll) - w[1]*rolld*np.sin(roll) - w[2]*rolld*np.cos(roll)
#    yawdd = (wd[1]*np.sin(roll) + wd[2]*np.cos(roll))/np.cos(pitch) + w[1]*(rolld*np.cos(roll)/np.cos(pitch) + pitchd*np.sin(roll)*np.sin(pitch)/(np.cos(pitch)**2)) + w[2]*(-rolld*np.sin(roll)/np.cos(roll) + pitchd*np.cos(roll)*np.sin(pitch)/(np.cos(pitch)**2))
#    return rolldd, pitchdd, yawdd
    
    
    
def get_plan(p, p1, p2):
    alpha, beta, gamma = p
    a, b, c = p1
    A, B, C = p2
    try:
        G = alpha-a - (gamma-c)/(gamma-C) * (alpha-A)
        H =  beta-b - (gamma-c)/(gamma-C) * (beta-B)
    except:
        if gamma-c == 0:
            try:
                if (alpha-A)/(alpha-a) != (beta-B)/(beta-b):
                    phi, theta, ksi = 0, 0, 1
                else:
                    return None         # Colinearity
            except:
                return None         # Colinearity
        else:
            try:
                G1 = alpha-a - (beta-b)/(beta-B) * (alpha-A)
                phi, ksi = 1, -G1 / (gamma-c)
                try:
                    H1 = beta-b - (alpha-a)/(alpha-A) * (beta-B)
                    if H1 != 0:
                        theta = -ksi*(gamma-c)/H1
                    else:
                        phi, theta, ksi = 0, 1, 0
                except:
                    if beta-B != 0:
                        theta = 0
                    else:
                        return None         # Confusion
            except:
                if alpha-A != 0:
                    phi = 0
                    theta, ksi = 1, -(beta-b)/(gamma-c)
                else:
                    return None         # Confusion
        d = -theta*b - ksi*c - phi*a
        return [phi, theta, ksi, d]
    try:
        J =  beta-b - (alpha-a)/(alpha-A) * (beta-B)
        K = gamma-c - (alpha-a)/(alpha-A) * (gamma-C)
    except:
        if alpha-a == 0:
            try:
                if (gamma-C)/(gamma-c) != (beta-B)/(beta-b):
                    phi, theta, ksi = 1, 0, 0
                else:
                    return None         # Colinearity
            except:
                return None         # Colinearity
        else:
            try:
                J1 = beta-b - (gamma-c)/(gamma-C) * (beta-B)
                phi, theta = -J1/(alpha-a), 1
                try:
                    K1 = gamma-c - (beta-b)/(beta-B) * (gamma-C)
                    if K1 != 0:
                        ksi = -phi*(alpha-a)/K1
                    else:
                        phi, theta, ksi = 0, 0, 1
                except:
                    if gamma-C != 0:
                        ksi = 0
                        d = -theta*b - ksi*c - phi*a
                        return [phi, theta, ksi, d]
                    else:
                        return None         # Confusion
            except:
                if beta-B != 0:
                    theta = 0
                    phi, ksi = -(gamma-c)/(alpha-a), 1
                else:
                    return None         # Confusion
        d = -theta*b - ksi*c - phi*a
        return [phi, theta, ksi, d]
        
    if H != 0 and K != 0:
        phi, theta, ksi = 1, -G/H, J*G/(H*K)
    elif K != 0:
        if G != 0:
            phi, theta, ksi = 0, 1, -J/K
        else:
            return None         # Colinearity
    elif H != 0:
        if J != 0:
            phi, theta, ksi = 1, 0, -(gamma-C)/(alpha-A)
        else:
            return None         # Colinearity
    else:
        if G == 0 or J == 0:
            return None         # Colinearity
        else:
            phi, theta, ksi = 0, 0, 1
    d = -theta*b - ksi*c - phi*a
    return [phi, theta, ksi, d]


def curve_center(p, p1, p2):
    alpha, beta, gamma = p
    a, b, c = p1
    A, B, C = p2
    try:
        phi, theta, ksi, d = get_plan(p, p1, p2)
    except:
        return None
    X = ((a+alpha)*(a-alpha) + (b+beta)*(b-beta) + (c+gamma)*(c-gamma)) / 2 
    Y = ((B+alpha)*(B-alpha) + (B+beta)*(B-beta) + (C+gamma)*(C-gamma)) / 2
    M = np.array([[a-alpha, b-beta, c-gamma], 
                  [A-alpha, B-beta, C-gamma], 
                  [    phi,  theta,     ksi]])
    im = np.array([X, Y, -d])
    return np.linalg.solve(M, im), [phi, theta, ksi, d]
    
    
def twist_center(p, p1, p2):
    '''As command twist is only on z-axis'''
    x, y, x1, y1, x2, y2 = p[:2], p1[:2], p2[:2]
    try:
        t = (x-x1)/(x2-x1)
        if y1 + (y2-y1)*t == y and z1 + (z2-z1)*t == z:
            return None
    except:
        try:
            t = (y-y1)/(y2-y1)
            if z1 + (z2-z1)*t == z:
                return None
        except:
            return None
    X = ((x1+x)*(x1-x) + (y1+y)*(y1-y)) / 2 
    Y = ((x2+x)*(x2-x) + (x2+y)*(x2-y)) / 2
    M = np.array([[x1-x, y1-y], 
                  [x2-x, y2-y]])
    im = np.array([X, Y])
    return np.linalg.solve(M, im)
    
    
def eq_twist(tc, pos, speed):
    if tc is None:
        return np.array([speed[0], speed[1], speed[2], 0, 0, 0])
    radius = np.sqrt(np.sum(np.square(tc-pos)))
    speed_norm = np.sqrt(np.sum(np.square(speed)))
    return np.array([speed[0], speed[1], speed[2], 0, 0, speed_norm/radius])
    

                        
def cross(u, v):
    assert(len(u) ==3 and len(v) == 3)
    uext, vext = np.array(list(u) + list(u)), np.array(list(v) + list(v))
    out = np.zeros(3)
    for i in range(1, 4):
        out[i-1] = uext[i]*vext[i+1] - uext[i+1]*vext[i]
    return out
