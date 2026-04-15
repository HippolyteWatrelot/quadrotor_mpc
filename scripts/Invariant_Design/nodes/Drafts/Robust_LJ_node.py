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

from std_msgs.msg import Float64MultiArray, Float64, MultiArrayDimension, Int16
from quadrotor_mpc.srv import Equilibrium
#from quadrotor_mpc.symbolic_state_to_input import control_backward
from quadrotor_mpc.forward_test import forward

from optim_utils.py import *


c_config = None
main_vars = ['x', 'xd', 'xdd', 'y', 'yd', 'ydd', 'z', 'zd', 'zdd', 'roll', 'rolld', 'rolldd', 'pitch', 'pitchd', 'pitchdd', 'yaw', 'yawd', 'yawdd']
quadrotor_vars = ['xq', 'd_xq', 'd2_xq', 'yq', 'd_yq', 'd2_yq', 'z', 'd_z', 'd2_z', 'roll', 'rolld', 'rolldd', 'pitch', 'pitchd', 'pitchdd', 'yaw', 'yawd', 'yawdd']
vars_dict = [Symbol(a) for a in X]
U = ['ux', 'uy', 'uz', 'uwz']
NX, nx = 18, 8
nu = 4
AD, Ad, Bd = np.zeros([NX, NX]), np.array([nx+6, nx+6]), np.array([nx+6, nu])
AS, BS = Matrix(nx, nx), Matrix(nx, nu)
A, B = np.zeros([nx, nx]), np.zeros([nx, nu])
Q, R = np.zeros([nx, nx]), np.zeros([nu, nu])
Qinv, Rinv = np.zeros([nx, nx]), np.zeros([nu, nu])
K = np.zeros([nu, nx])
F = None
dt = 0.1
g = 9.8065


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
theta = [MX.sym('I00'), MX.sym('I11'), MX.sym('I22')]

input_vars = ['d_xq', 'd_yq', 'd_zq', 'roll', 'rolld', 'pitch', 'pitchd', 'yawd']
output_vars = ['d2_xq', 'd2_yq', 'd2_z', 'rolld', 'rolldd', 'pitchd', 'pitchdd', 'yawdd']
direct_outputs = ['d2_xq', 'd2_yq', 'd2_z', 'rolldd', 'pitchdd', 'yawdd']

sympy_input_vars = Matrix([Symbol(a) for a in input_vars])
sympy_output_vars = Matrix([Symbol(a) for a in output_vars])

xq, d_xq, d2_xq, yq, d_yq, d2_yq, z, d_z, d2_z, roll, rolld, rolldd, pitch, pitchd, pitchdd, yaw, yawd, yawdd = [MX.sym('xq'), MX.sym('d_xq'), MX.sym('d2_xq'), 
                                                                                                                 MX.sym('yq'), MX.sym('d_yq'), MX.sym('d2_yq'), 
                                                                                                                 MX.sym('z'), MX.sym('d_z'), MX.sym('d2_z'), 
                                                                                                                 MX.sym('roll'), MX.sym('rolld'), MX.sym('rolldd'),
                                                                                                                 MX.sym('pitch'), MX.sym('pitchd'), MX.sym('pitchdd'), 
                                                                                                                 MX.sym('yaw'), MX.sym('yawd'), MX.sym('yawdd')]
                                                                                           
ca_input_vars = vertcat(*[d_xq, d_yq, d_z, roll, rolld, pitch, pitchd, yawd])
ca_output_vars = vertcat(*[d2_xq, d2_yq, d2_z, rolld, rolldd, pitchd, pitchdd, yawdd])





# Optim casadi function
def find_alpha(A, B, K, P, k, F, G, state, init_alpha=5, tol=0.01, max_iter=10):
    global lim_inputs
    global ca_input_vars, ca_output_vars
    global theta, use_Interval
    '''phi(x,θ) = f(x,θ,Kx) - (A + BK)x'''
    '''Search max {x.T @ P @ phi(x,θ) - k * x.T @ P @ x,  x.T @ P @ x < alpha}'''
    _A, _B, _K, _P, _F, _G = ca.MX(A), ca.MX(B), ca.DM(K), ca.DM(P), ca.DM(F), ca.DM(G)
    inv_P = ca.inv(_P)
    if not use_Interval:
        phi = func(ca_input_vars, _K*ca_input_vars) - (_A* + _B*_K)*ca_input_vars                   # phi symbolic function forward
        _f = k*ca_input_vars.T()*_P*ca_input_vars - ca_input_vars.T()*_P*phi                        # k * x.T @ P @ x - x.T @ P @ phi(x)
        f_cost = ca.Function('f', [ca_input_vars], [_f])  # function to be minimized
    else:
        phi = func(ca_input_vars, _K*ca_input_vars, theta) - (_A* + _B*_K)*ca_input_vars          # phi symbolic function forward
        _f = k*ca_input_vars.T()*_P*ca_input_vars - ca_input_vars.T()*_P*phi                      # k * x.T @ P @ x - x.T @ P @ phi(x,θ)
        f_cost = ca.Function('f', [ca_input_vars, theta], [_f])  # function to be minimized
        
    # Constraints
    ellipsoid_constraint = mtimes(ca_input_vars.T(), mtimes(_P, ca_input_vars))                # x.T @ P @ x < alpha
    g = [f_cost, ellipsoid_constraint]                                                         # function to be minimized is also constrained to be positive
    lbg, ubg = [0, 0], [ca.inf, alpha]                                                         # Ellipsoid bounds to be found
    if use_Interval:
        for i, key in enumerate(Interval.keys()):
            g.append(theta[i])
            lbg.append(Interval[key][0])
            ubg.append(Interval[key][1])
    
    # Solver
    nlp = {'x': ca_input_vars, 'f': f_cost, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp)                                                    # NLP Solver
    
    # Init      
    previous_alpha, alpha = ca.inf, init_alpha
    x0 = MX(state)
    prev_sol, it = None, 0
            
    while True:                                                                                # Dichotomic Search
        sol = solver(x0=x0, lbg=lbg, ubg=ubg)
        if sol['stats'] == 'Solve_Succeeded' and ellipsoid_linear_constraints_inclusion(state, inv_P, alpha, _F, _G, _K):
            if np.abs(alpha-previous_alpha)<tol or it>=max_iter:
                if np.abs(alpha-previous_alpha)>tol:
                    return None
                break
            a = alpha
            if not prev_sol:
                alpha += np.abs(alpha-previous_alpha) / 2
            else:
                alpha += np.abs(alpha-previous_alpha)
            previous_alpha = a
            prev_sol = True
        else:
            a = alpha
            if prev_sol:
                alpha -= (previous_alpha-alpha) / 2
            else:
                alpha -= (previous_alpha-alpha)
            previous_alpha = a
            prev_sol = False
        ubg[1] = alpha
        it += 1
    return alpha




### PROCEDURE

def traj_linear_matrices(req):
    global AS, BS, A, B, Q, R, K, F, config, X, U, nx, nu, dt, g
    xd, yd, zd, yawd = req.xd, req.yd, req.zd, req.yawd
    target_x, target_y, target_z, target_yaw = req.target_x, req.target_y, req.target_z, req.target_yaw      # target pos and yaw at requested equilibrium state
    F = req.linear_constraints
    T1, T2 = build_transition_mats(target_yaw)
    d_xq, d_yq, d_z = xd*np.cos(target_yaw)+yd*np.sin(target_yaw), -xd*np.sin(target_yaw)+yd*np.cos(target_yaw), zd
    xdd, ydd = -yawd*yd, yawd*xd       # induced acceleration from speed considering equilibrium (in quadrotor frame, they are 0)
    roll, rolld = -np.atan2(yawd*(xd*np.cos(target_yaw) + yd*np.sin(target_yaw)), g), 0
    pitch, pitchd = -np.atan2(yawd*(yd*np.cos(target_yaw) - xd*np.sin(target_yaw)), g), 0   # roll and pitch values in considered equilibrium (0 if straight)
    eq_values = {
                 'd_xq': d_xq                                                                                           # simple offset
                 'd_yq': d_yq                                                                                           # simple offset 
                 'd_z': d_z,                                                                                            # same as d_z
                 'roll': roll,                                                                                          # function of xd, yd, yawd and target yaw
                 'rolld': rolld,
                 'pitch': pitch,                                                                                        # function of xd, yd, yawd and target yaw
                 'pitchd': pitchd,
                 'yawd': yawd, 
                 'ux': d_xq,                                                                                            # Obvious equilibrium input control
                 'uy': d_yq,                                                                                            # Obvious equilibrium input control
                 'uz': d_z,                                                                                             # Obvious equilibrium input control
                 'uwz': yawd                                                                                            # Obvious equilibrium input control
                }
            
    if not use_Interval:
        for i in range(nx):
            for j in range(nx):
                A[i, j] = AS[i, j].subs(eq_values).evalf()
            for j in range(nu):
                B[i, j] = BS[i, j].subs(eq_values).evalf()
        K = continuous_Riccati(A, B, Q, R)
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
    
    AK = A + B @ K    
    lambda_max = np.max(np.linalg.eig(AK))             # Must be negative
    try:
        assert lambda_max < 0
    except:
        print('UNSTABLE ClOSED LOOP LINEAR APPROXIMATION')
        return False
    k = -0.95*lambda_max                                        # (Must be > 0), prevents discrete approximation imprecisions
    P = continuous_Lyapunov(A, B, K, Q, R, k)                          # <-------------------------------------------------
    alpha = find_alpha(A, B, K, P, k, F, G, [eq_values[key] for key in input_vars])     # <--------------------------------- C&A ALGORITHM
        
    AK_msg, P_msg = Float64MultiArray(), Float64MultiArray()
    AK_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
    P_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
    AK_msg.layout.dim[0].size = nx
    AK_msg.layout.dim[1].size = nx
    P_msg.layout.dim[0].size = nx
    P_msg.layout.dim[1].size = nx
    discrete_AK = scipy.signal.expm(AK*dt)            # Discrete A+BK
    for i in range(nx):
        for j in range(nx):
            AK_msg.append(discrete_AK[i, j])
            P_msg.data.append(P[i, j])
    return AK_msg, P_msg, alpha          # Returns Discrete A+BK, invariant-dimension ellipsoid P and its corresponding attraction-radius alpha
    




### INITIALIZING NODE
    
    
def build_continuous_symbol_matrices():
    global AS, BS, Q, R, Qinv, Rinv, c_config, input_vars, output_vars
    for i, ov in enumerate(output_vars):
        for j, iv in enumerate(input_vars):
            AS[i, j] = sympify(c_config[ov][iv])
        for j, c in enumerate(U):
            BS[i, j] = sympify(c_config[ov][c])
    WQ = list(rospy.get_param('Qnl'))
    Qvec = WQ[1:9:3] + WQ[9:11] + WQ[12:14] + [WQ[16]]
    R_vec = np.remove(rospy.get_param('Rnl'), [2*i for i in range(4)])
    Q = np.diag(Q_vec)
    R = np.diag(R_vec)
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
        der_file = "yaml/Linear_Coefficients/INTERVAL_reduced_partial_derivatives_values.yaml"
        acc_file = "yaml/INTERVAL_accelerations_formulas.yaml"
        f = ca.Function.load('reduced_dyn_f_INTERVAL.casadi')
    else:
        der_file = "yaml/Linear_Coefficients/reduced_partial_derivatives_values.yaml"
        acc_file = "yaml/accelerations_formulas.yaml"
        f = ca.Function.load('reduced_dyn_f.casadi')
    
    with open(der_file, "r") as f:
        d_config = yaml.safe_load(f)
    c_config = {key: d_config[key] for key in direct_outputs}
    with open(acc_file, "r") as g:
        acc_formulas = yaml.safe_load(g)
            
    build_continuous_symbol_matrices()
    main_node()
