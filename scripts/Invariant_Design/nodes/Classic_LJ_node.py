#! /usr/bin/env python3

import numpy as np
import cvxpy as cp
import random
import rospy
import sys
import os
from sympy import *
import casadi as ca
import scipy
import yaml
from math import atan2

from std_msgs.msg import Float64MultiArray, Float64, MultiArrayDimension, Int16
from quadrotor_mpc.srv import Equilibrium
#from quadrotor_mpc.symbolic_state_to_input import control_backward
#from quadrotor_mpc.forward_test import forward

from quadrotor_mpc.optim_utils import *
from quadrotor_mpc.transform_utils import eulerd2w


c_config = None
main_vars = ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd', 'roll', 'rolld', 'rolldd', 'pitch', 'pitchd', 'pitchdd', 'yaw', 'yawd', 'yawdd']
quadrotor_vars = ['xq', 'd_xq', 'd2_xq', 'yq', 'd_yq', 'd2_yq', 'z', 'd_z', 'd2_z', 'roll', 'rolld', 'rolldd', 'pitch', 'pitchd', 'pitchdd', 'yaw', 'yawd', 'yawdd']
#vars_dict = [Symbol(a) for a in X]
U = ['ux', 'uy', 'uz', 'uwz']
nx = 12
nu = 4
AS, BS = Matrix(np.zeros([nx, nx])), Matrix(np.zeros([nx, nu]))
AdS, BdS = Matrix(np.zeros([nx, nx])), Matrix(np.zeros([nx, nu]))
A, B = np.zeros([nx, nx]), np.zeros([nx, nu])
Q, R = np.zeros([nx, nx]), np.zeros([nu, nu])
Qinv, Rinv = np.zeros([nx, nx]), np.zeros([nu, nu])
K = np.zeros([nu, nx])
dt = 0.01 #0.1
g = 9.8065
dtS = Symbol('dt')


I = np.array([[0.0115202,         0,         0],
              [        0, 0.0115457,         0],
              [        0,         0, 0.0218256]])
I1, I2, I3 = I[0, 0], I[1, 1], I[2, 2]
mass = 1.478
mass_offset = 0.05
Intervals = {'I00':  [I1 - I1*mass_offset/mass, I1 + I1*mass_offset/mass], 
             'I11':  [I2 - I2*mass_offset/mass, I2 + I2*mass_offset/mass], 
             'I22':  [I3 - I3*mass_offset/mass, I3 + I3*mass_offset/mass], 
             'mass': [mass-mass_offset, mass+mass_offset]}
use_Interval = False
I00, I11, I22 = symbols('I00 I11 I22')
theta = [ca.MX.sym('I00'), ca.MX.sym('I11'), ca.MX.sym('I22')]

input_vars = ['d_xq', 'd_yq', 'd_z', 'roll', 'rolld', 'pitch', 'pitchd', 'yawd', 'prev_uwx_b', 'prev_uwy_b']
output_vars = ['d2_xq', 'd2_yq', 'd2_z', 'rolld', 'rolldd', 'pitchd', 'pitchdd', 'yawdd', 'd_prev_uwx_b', 'd_prev_uwy_b']
direct_outputs = ['d2_xq', 'd2_yq', 'd2_z', 'rolldd', 'pitchdd', 'yawdd']

sympy_input_vars = Matrix([Symbol(a) for a in input_vars])
sympy_output_vars = Matrix([Symbol(a) for a in output_vars])

xq, d_xq, d2_xq, yq, d_yq, d2_yq, z, d_z, d2_z, roll, rolld, rolldd, pitch, pitchd, pitchdd, yaw, yawd, yawdd = [ca.MX.sym('xq'), ca.MX.sym('d_xq'), ca.MX.sym('d2_xq'), 
                                                                                                                 ca.MX.sym('yq'), ca.MX.sym('d_yq'), ca.MX.sym('d2_yq'), 
                                                                                                                 ca.MX.sym('z'), ca.MX.sym('d_z'), ca.MX.sym('d2_z'), 
                                                                                                                 ca.MX.sym('roll'), ca.MX.sym('rolld'), ca.MX.sym('rolldd'),
                                                                                                                 ca.MX.sym('pitch'), ca.MX.sym('pitchd'), ca.MX.sym('pitchdd'),
                                                                                                                 ca.MX.sym('yaw'), ca.MX.sym('yawd'), ca.MX.sym('yawdd')]
 
prev_uwx_b, prev_uwy_b, d_prev_uwx_b, d_prev_uwy_b = [ca.MX.sym('prev_uwx_b'), ca.MX.sym('prev_uwy_b'),
                                                      ca.MX.sym('d_prev_uwx_b'), ca.MX.sym('d_prev_uwy_b')]
                                                                                           
ca_input_vars = ca.vertcat(*[d_xq, d_yq, d_z, roll, rolld, pitch, pitchd, yawd, prev_uwx_b, prev_uwy_b])
ca_output_vars = ca.vertcat(*[d2_xq, d2_yq, d2_z, rolld, rolldd, pitchd, pitchdd, yawdd, d_prev_uwx_b, d_prev_uwy_b])





# Optim casadi function
def find_alpha(AK, K, P, k, F, G, xeq, ueq, init_alpha=0.05, tol=0.01, max_iter=100):
    global lim_inputs
    global ca_input_vars, ca_output_vars
    global theta, use_Interval
    global func
    global dt, Q, R
    '''phi(x,θ) = f(x,θ,ueq + K*(x-xeq)) - (A + BK)*(x-xeq)'''
    '''Search max {(x-xeq).T @ P @ phi(x,θ) - k * (x-xeq).T @ P @ (x-xeq),  (x-xeq).T @ P @ (x-xeq) < alpha}'''
    
    AK, K, P, F, G = ca.DM(AK), ca.DM(K), ca.DM(P), ca.DM(F), ca.DM(G)
    inv_P = ca.inv(P)
    x = ca_input_vars
    rel_x = x - xeq
    if not use_Interval:
        phi = func(x, ueq + ca.mtimes(K, rel_x)) - ca.mtimes(AK, rel_x)                                # phi symbolic function forward (differential)
        _f = k*ca.mtimes(rel_x.T, ca.mtimes(P, rel_x)) - ca.mtimes(rel_x.T, ca.mtimes(P, phi))         # k * rel_x.T @ P @ rel_x - rel_x.T @ P @ phi(x)
        f_cost = ca.Function('f', [x], [_f])  # function to be minimized
    else:
        phi = func(x, ueq + ca.mtimes(K, rel_x), theta) - ca.mtimes(AK, rel_x)        # phi symbolic function forward
        _f = k*ca.mtimes(rel_x.T, ca.mtimes(P, rel_x)) - ca.mtimes(rel_x.T, ca.mtimes(P, phi))         # k * x.T @ P @ x - x.T @ P @ phi(x,θ)
        f_cost = ca.Function('f', [x, theta], [_f])  # function to be minimized
    
    near_state = xeq + ca.vertcat(*[0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0])
    print('near_state: ', near_state)
    print(f_cost(near_state))
    print('FORWARD: ', func(ca.vertcat(*[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), ca.vertcat(*[2, 0, 0, 0])))
    s = near_state
    for i in range(1000):
        #s = s + func(s, ca.mtimes(K, s))*dt
        s = s + func(s, ca.vertcat([1, 0, 0, 0]))*dt
        print(s)
    return
        
    # Constraints
    ellipsoid_constraint = ca.mtimes(rel_x.T, ca.mtimes(P, rel_x))                # (x-xeq).T @ P @ (x-xeq) < alpha
    g = [f_cost(ca_input_vars), ellipsoid_constraint]                                                                  # function to be minimized is also constrained to be positive
    lbg, ubg = [0, 0], [ca.inf, init_alpha]                                                                           # Ellipsoid bounds to be found
    if use_Interval:
        for i, key in enumerate(Interval.keys()):
            g.append(theta[i])
            lbg.append(Interval[key][0])
            ubg.append(Interval[key][1])
    
    # Solver
    print('g[0]:\n', g[0])
    print('g[1]:\n', g[1])
    nlp = {'x': ca_input_vars, 'f': f_cost(ca_input_vars), 'g': ca.vertcat(*g)}
    opts = {
      'ipopt.hessian_approximation': 'limited-memory',
      'ipopt.limited_memory_max_history': 20,
      'ipopt.mu_strategy': 'adaptive',
      'ipopt.linear_solver': 'mumps',
      'ipopt.nlp_scaling_method': 'gradient-based',
      'ipopt.tol': 1e-6,
      'ipopt.acceptable_tol': 1e-4,
      'ipopt.max_iter': 1000,
      'ipopt.warm_start_init_point': 'yes',
      'print_time': 0, 'ipopt.print_level': 0
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp)#, opts)                                                    # NLP Solve
    
    # Init      
    alpha = init_alpha
    x0 = ca.DM(xeq)
    prev_sol, it = None, 0
    eps = alpha/2
            
    while True:                
        print('lbg: ', lbg)
        print('ubg: ', ubg)                                                                # Dichotomic Search
        sol = solver(x0=x0, lbg=lbg, ubg=ubg)
        if not solver.stats().get('success', False): #and ellipsoid_linear_constraints_inclusion(state, inv_P, alpha, F, G, K):
            print("success")    # func min is positive
            if eps<tol or it>=max_iter:
                if eps>tol:
                    return None
                break
            a = alpha
            if prev_sol:
                eps = alpha/2
            alpha += eps
            previous_alpha = a
            prev_sol = True
        else:
            print("failure")
            a = alpha
            if not prev_sol:
                eps = alpha/2
            alpha -= eps
            previous_alpha = a
            prev_sol = False
        ubg[1] = alpha
        it += 1
    return alpha




### PROCEDURE

def traj_linear_matrices(req):
    global AS, BS, A, B, Q, R, K, F, config, X, U, nx, nu, dt, g
    xd, yd, zd, yawd = req.vars
    target_x, target_y, target_z, target_yaw = req.targets      # target pos and yaw at requested equilibrium state
    F, G = np.array(req.F).reshape([2*nx, nx]), np.array(req.G).reshape([2*nu, nx])
    #T1, T2 = build_transition_mats(target_yaw)
    d_xq, d_yq, d_z = xd*np.cos(target_yaw)+yd*np.sin(target_yaw), -xd*np.sin(target_yaw)+yd*np.cos(target_yaw), zd
    xdd, ydd = -yawd*yd, yawd*xd       # induced acceleration from speed considering equilibrium (in quadrotor frame, they are 0)
    roll, rolld = -atan2(yawd*(xd*np.cos(target_yaw) + yd*np.sin(target_yaw)), g), 0
    pitch, pitchd = -atan2(yawd*(-xd*np.sin(target_yaw) + yd*np.cos(target_yaw)), g), 0   # roll and pitch values in considered equilibrium (0 if straight)
    eq_values = {
                 'd_xq': d_xq,                                                                                          # simple offset
                 'd_yq': d_yq,                                                                                          # simple offset 
                 'd_z': d_z,                                                                                            # same as d_z
                 'roll': roll,                                                                                          # function of xd, yd, yawd and target yaw
                 'rolld': rolld,
                 'pitch': pitch,                                                                                        # function of xd, yd, yawd and target yaw
                 'pitchd': pitchd,
                 'yawd': yawd, 
                 'ux': d_xq,                                                                                            # Obvious equilibrium input control
                 'uy': d_yq,                                                                                            # Obvious equilibrium input control
                 'uz': d_z,                                                                                             # Obvious equilibrium input control
                 'uwz': yawd,                                                                                           # Obvious equilibrium input control
                 'prev_uwx_b': 0,
                 'prev_uwy_b': 0
                }
    xeq = ca.vertcat(*[eq_values[key] for key in input_vars])
    ueq = ca.vertcat(*[d_xq, d_yq, d_z, yawd])
    print('xeq: ', xeq)
    print('ueq: ', ueq)
    if not use_Interval:
        for i in range(nx):
            for j in range(nx):
                try:
                    A[i, j] = AS[i, j].subs(eq_values).evalf()
                except:
                    A[i, j] = AS[i, j]
            for j in range(nu):
                try:
                    B[i, j] = BS[i, j].subs(eq_values).evalf()
                except:
                    B[i, j] = BS[i, j]
        print(f'A:\n{A}\n')
        print(f'B:\n{B}\n')
        K = continuous_Riccati(A, B, Q, R)
        #K = discrete_Riccati(A, B, Q, R)
        print(f'K:\n{K}\n')
    else:
        for i in range(nx):
            for j in range(nx):
                symb_value = AS[i, j].subs(eq_values).evalf()                       # Still symbolic because of intervalss I1, I2, I3
                fl = lambdify((I00, I11, I22), symb_value, 'numpy')
                A[i, j] = fl(theta[0], theta[1], theta[2])                          # Symbolic
            for j in range(nu):
                symb_value = BS[i, j].subs(eq_values).evalf()
                fl = lambdify((I00, I11, I22), symb_value, 'numpy')
                B[i, j] = fl(theta[0], theta[1], theta[2])                          # Symbolic
        K = continuous_convex_LMI_process(A, B, Qinv, Rinv)            # <------------------------------------------------- approximates optimal K on convex polytope defined by intervals
        A = np.array(Matrix(A).subs({'I00': I1, 'I11': I2, 'I22': I3}).evalf())     # mid Intervals A
        B = np.array(Matrix(B).subs({'I00': I1, 'I11': I2, 'I22': I3}).evalf())     # mid Intervals B
        #Ad, Bd = cont2dis(A, B, dt, order=1)
    
    continuous_AK = A + B @ K
    print('A+BK:\n', continuous_AK)
    print('A eigenvalues:\n', np.linalg.eig(A)[0])
    print('A+BK eigenvalues:\n', np.linalg.eig(continuous_AK)[0])
    lambda_max = np.max(np.linalg.eig(continuous_AK)[0])
    try:
        assert lambda_max.real < 0
    except:
        print('UNSTABLE ClOSED LOOP LINEAR APPROXIMATION')
        return False
    k = -lambda_max*0.95
    print("k: ", k)
    print('k*(A+BK) eigenvalues:\n', np.linalg.eig(k*continuous_AK)[0])
    P = continuous_Lyapunov(continuous_AK, K, Q, R, k=k)       # To have a more restrictive P
    print(f'P:\n{P}')
    print(Q, '\n', R)
    x = ca.vertcat(*[0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #print((x-xeq).T * ca.DM(P) * (x-xeq))
    alpha = find_alpha(A, B, K, P, k, F, G, xeq, ueq)     # <--------------------------------- C&A ALGORITHM
    discrete_AK = scipy.linalg.expm(continuous_AK*dt)
    
    return True
    




### INITIALIZING NODE
    
    
def build_continuous_symbol_matrices():
    global AS, BS, Q, R, Qinv, Rinv, c_config, input_vars, output_vars
    #print(c_config.keys())
    for i, ov in enumerate(output_vars):
        for j, iv in enumerate(input_vars):
            #print(ov, iv)
            AS[i, j] = sympify(c_config[ov][iv])
        for j, c in enumerate(U):
            BS[i, j] = sympify(c_config[ov][c])
    WQ = list(rospy.get_param('Qnl'))
    Qvec = WQ[1:9:3] + WQ[9:11] + WQ[12:14] + [WQ[16]]
    indexes = [2*i+1 for i in range(4)]
    Rvec_load = rospy.get_param('Rnl')
    Rvec = [Rvec_load[i] for i in indexes]
    #Q = np.diag(Qvec)
    Q = np.eye(nx)
    R = np.diag(Rvec)
    Qinv, Rinv = np.linalg.inv(Q), np.linalg.inv(R)
    
    
def build_discrete_symbol_matrices():
    global AdS, BdS, Q, R, Qinv, Rinv, c_config, input_vars, output_vars, dtS, dt
    for i, ov in enumerate(output_vars):
        for j, iv in enumerate(input_vars):
            #_id = lambda x : x + 1 if i == j else x
            try:
                AdS[i, j] = dt*sympify(c_config[ov][iv])
                #AdS[i, j] = AdS[i, j].subs({'dtS': dt}).evalf()
                if i == j:
                    AdS[i, j] += 1
                    print(type(AdS[i, j]))
            except:
                pass
        for j, c in enumerate(U):
            try:
                BdS[i, j] = dt*sympify(c_config[ov][c])
                #BdS[i, j] = BdS[i, j].subs({'dtS': dt}).evalf()
            except:
                pass
    Q, R = np.diag(list(rospy.get_param('Q'))), np.diag(list(rospy.get_param('R')))
    Qinv, Rinv = np.linalg.inv(Q), np.linalg.inv(R)
    

def main_node():
    rospy.init_node("linear_matrices")
    s_traj = rospy.Service("linear_matrices", Equilibrium, traj_linear_matrices)
    rospy.spin()

        
if __name__ == "__main__":

    try:
        use_Interval = bool(int(sys.argv[1]))
    except:
        use_Interval = False
        
    if use_Interval:
        der_file = "/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/yaml/Linear_Coefficients/INTERVAL_reduced_partial_derivatives_values.yaml"
        acc_file = "/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/yaml/Linear_Coefficients/INTERVAL_reduced_accelerations_formulas.yaml"
        func = ca.Function.load('/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/Invariant_Design/CasADi_formulas/reduced_dyn_f_INTERVAL.casadi')
    else:
        der_file = "/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/yaml/Linear_Coefficients/reduced_partial_derivatives_values.yaml"
        acc_file = "/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/yaml/Linear_Coefficients/reduced_accelerations_formulas.yaml"
        func = ca.Function.load('/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/Invariant_Design/CasADi_formulas/reduced_dyn_f.casadi')
    
    with open(der_file, "r") as f1:
        d_config = yaml.safe_load(f1)
    c_config = {key: d_config[key] for key in pre_output_vars}
    with open(acc_file, "r") as f2:
        acc_formulas = yaml.safe_load(f2)
            
    #build_continuous_symbol_matrices()
    build_discrete_symbol_matrices()
    main_node()
