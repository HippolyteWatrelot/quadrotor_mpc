#! /usr/bin/env python3

import numpy as np
import casadi as ca
import cvxpy as cp
import scipy
from intvalpy import Interval
from sympy import sympify, lambdify

from sage.all import Polyhedron, vector



def cont2dis(A, B, dt, order):
    nx, nu = A.shape[1], B.shape[1]
    if order is None:
        AB = scipy.linalg.expm(np.block([[A, B], [np.zeros([nu, nx]), np.zeros([nu, nu])]])*dt)
        Ad, Bd = AB[:nx, :nx], AB[:nx, nx:]
    else:
        Ad, Bd = A*dt + np.eye(nx), B*dt
    return Ad, Bd


def continuous_Riccati(A, B, Q, R):
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = -np.linalg.inv(R) @ B.T @ P
    return K
    
    
def continuous_Lyapunov(AK, K, Q, R, k=0):
    # k E [0, -Lambda_max(A+BK)[
    nx = A.shape[0]
    Q_star = -(Q + K.T @ R @ K)
    P = scipy.linalg.solve_continuous_lyapunov((AK + k*np.eye(nx)), Q_star)
    return P
    
    
def discrete_Riccati(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    return K
    
    
def discrete_Lyapunov(AK, K, Q, R, k=1):
    Q_star = Q + K.T @ R @ K
    P = scipy.linalg.solve_discrete_lyapunov(k*AK.T, Q_star)
    return P
    
    
def continuous_convex_LMI_process(A, B, Qinv, Rinv, Intervals, solver=cp.MOSEK, threshold=1e-6):
    # Interval Mode
    # TO BE MODIFIED (McCORMICK)
    nx, nu = A.shape[0], B.shape[1]
    N = len(Intervals)
    A_list = [np.array(Matrix(A).subs({key: Intervals[key][i//(2**n)%2] for n, key in enumerate(Intervals.keys())}).evalf()) for i in range(2**N)]      # FAKE Polytope
    B_list = [np.array(Matrix(B).subs({key: Intervals[key][i//(2**n)%2] for n, key in enumerate(Intervals.keys())}).evalf()) for i in range(2**N)]
    X = cp.Variable((nx, nx), symmetric=True)
    Y = cp.Variable((nu, nx))
    constraints = [X >> 1e-6 * np.eye(nx)]  # X ≻ 0    
    for A, B in zip(A_list, B_list):
        M = A @ X + B @ Y
        # Schur Complements
        LMI = cp.bmat([[M + M.T,                   X,                Y.T],
                       [      X,               -Qinv, np.zeros((nx, nu))],
                       [      Y,  np.zeros((nu, nx)),              -Rinv]])
        constraints.append(LMI << -1e-6 * np.eye(2*nx+nu))  # LMI ≺ 0
    #constraints.append(k*X >> m*(M + M.T)/2)                                         # Convex majoration (suboptimal)
    #obj = cp.Minimize(alpha * cp.sum_squares(Y) - beta * cp.log_det(X))
    #prob = cp.Problem(cp.Minimize(0), constraints)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver)
    assert prob.status in ["optimal", "optimal_inaccurate"]
    X_val = X.value
    Y_val = Y.value
    P = np.linalg.inv(X_val)
    K = Y_val @ P
    return K




# CASADI

def ellipsoid_linear_constraints_inclusion(x, P_inv, alpha, F, G, K, hs=0):
    C = []
    const_mat = F + ca.mtimes(G, K)          # must be < 1 dim nc
    nc = F.size()[0]
    for i in range(nc):
        ci = -const_mat[i, :].T
        C.append(ca.mtimes(ci.T, (x - ca.sqrt(alpha)*ca.mtimes(P_inv, ci)) / ca.sqrt(ca.mtimes(ci.T, ca.mtimes(P_inv, ci)))))
    comp = np.array(C) > -np.ones(nc) + hs
    return bool(np.prod(comp.astype(int)))
    

# CVXPY    
def discrete_Robust_Ke(A, B, D, Q, R, solver=cp.MOSEK, threshold=1e-6):
    # Tube MPC, Analog to discrete Riccati
    #Ad, Bd, Dd = cont2dis(A, B, dt, order=1), D*dt
    nx, nu, nw = A.shape[1], B.shape[1], D.shape[1]
    X = cp.Variable((nx, nx), symmetric=True)
    Y = cp.Variable((nu, nx))
    sq_gamma = cp.Variable(nonneg=True)
    M = A @ X + B @ Y
    constraints = [X >> threshold * cp.Constant(np.eye(nx))]  # X ≻ 0
    constraints.append(sq_gamma <= 10)
    block1 = np.block([[              X                ,  cp.Constant(np.zeros([nx, nw]))], 
                       [cp.Constant(np.zeros([nw, nx])),        sq_gamma*np.eye(nw)      ]])
    block2 = np.block([M, D])
    block3 = np.block([[np.sqrt(Q) @ X, cp.Constant(np.zeros([nx, nw]))], 
                       [np.sqrt(R) @ Y, cp.Constant(np.zeros([nu, nw]))]])   # As Q and R are diagonal and ≻ 0
    LMI = cp.bmat([[block1,              block2.T             ,              block3.T             ], 
                   [block2,                 X                 , cp.Constant(np.zeros([nx, nx+nu]))], 
                   [block3, cp.Constant(np.zeros([nx+nu, nx])),            np.eye(nx+nu)          ]])
    constraints.append(LMI >> threshold * cp.Constant(np.eye(3*nx+nu+nw)))  # LMI ≻ 0
    prob = cp.Problem(cp.Minimize(sq_gamma), constraints)
    prob.solve(solver=solver)
    assert prob.status in ["optimal", "optimal_inaccurate"]
    X_val = X.value
    Y_val = Y.value
    P = np.linalg.inv(X_val)
    Ke = Y_val @ P
    return Ke


def find_rho(ph1, ph2):
    inv_rhos = []
    for vert in ph2.vertices_list():
        inv_rho = None # max inv_rho_i: ∃ (λ0, ..., λn) ∈ ℝ^n / sum(λi)=1, inv_rho_i*vert = sum(λi*verts1(i))
        inv_rhos.append(inv_rho)
    return 1 / np.min(inv_rhos)


#def bounding_box(ph):
#    verts = ph.vertices_list()
#    n = len(verts[0])
#    box = IntervalVector(n)
#    for i in range(n):
#        box[i] = Interval(np.min(verts[:, i]), np.max(verts[:, i]))
#    return box


def IntervalMatrix(symbmat, bounds):
    # forward interval
    symbols = bounds.keys()
    for i in range(symbmath.shape[0]):
        for j in range(symbmath.shape[1]):
            f = lambdify(symbols, symbmath[i, j])
            interval_matrix[i, j] = f(bounds)
    return interval_matrix


def Residual_Jacobian(AK, AS, BS, K, bounds, sub_polyhedron):
    AKS = AS + BS*Matrix(K)
    box_AK = IntervalMatrix(AKS, bounds)     # Symbolic Matrix and bounds
    return sub_polyhedron.linear_transform(box_AK - AK)
    
    
def Polyhedron_calculation(AKe, AS, BS, Ke, D, W, F_dict, iters=10):
    main_polyhedron = W.linear_transform(D) # state variation form
    r_sum = main_polyhedron
    sub_polyhedron = main_polyhedron
    for _ in range(iters):
        RJac = Residual_Jacobian(AKe, AS, BS, Ke, F_dict, sub_polyhedron)
        sub_polyhedron = sub_polyhedron.linear_transform(AKe) + main_polyhedron
        r_sum = r_sum + RJac + sub_polyhedron
    rho = find_rho(main_polyhedron, r_sum)
    end_polyhedron_approx = r_sum / (1-rho)
    #Vs = bounding_box(end_polyhedron_approx)
    # Or an ellipsoid !!!
    return end_polyhedron_approx


def Polyhedron2Matrix(P, nx):
    ineqs = P.inequalities()
    n = len(ineqs)
    Vs = np.zeros([n, nx])
    for i in range(n):
        v = ineqs[i].vector()
        Vs[i, :] = -v[:-1] / v[-1]    # V*x <= 1 form
    return Vs


def hS_calculation(FKe, P):     # FKe = F+G*Ke
    nc, nx = FKe.shape
    Vs = Polyhedron2Matrix(P, nx)
    hs = np.zeros(nc)
    for i in range(nc):
        # max((F+GKe)i*x) : Vs*x < 1
        hs[i] = scipy.optimize.linprog(-FKe[i, :], Vs, 1)        # hS = max{e ∈ S}(F+GKe)e.
    return hs





### NonLinear MPC Tube control & validation


def states_sequence_linearization(states, inputs, state_keys, input_keys, partial_derivatives, dt):
    A_list, B_list = [], []
    d_states = partial_derivatives.keys()
    for n in range(len(states)):            # Horizon
        values = states[n] + inputs[n]
        keys = state_keys + input_keys
        dict_values = {key: values[i] for i, key in enumerate(keys)}
        for i, v in enumerate(d_states.keys()):
            for j, t in enumerate(states_keys):
                A[i, j] = dt*sympify(partial_derivatives[v][t]).subs(dict_values).evalf()
                if i == j:
                    A[i, j] += 1
            for j, t in enumerate(inputs_keys):
                B[i, j] = dt*sympify(partial_derivatives[v][t]).subs(dict_values).evalf()
        A_list.append(A)
        B_list.append(B)
    return A_list, B_list


def LTV_LQR(AB_list, KN, P, Q, R, N):
    K_list = []
    K_list.append(KN)
    for A, B in np.flip(AB_list, axis=0):
        S = R + B.T @ P @ B
        K_list.append(-S @ B.T @ P @ A)             # Recursive K
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(S) @ B.T @ P @ A
    return np.flip(K_list, axis=0).tolist()
    
    
def variable_tube(AB_list, K_list, AS, BS, state_noise, init_e, bounds):
    # state_noise is D * W
    e = init_e # + state_noise
    tube = [e]
    for A, B, K in np.concatenate((AB_list, K_list), axis=1):
        AK = A + B @ K
        RJac = Residual_Jacobian(AK, AS, BS, K, bounds, e)
        e = e.linear_transform(AK) + RJac + state_noise    # Tube sections
        tube.append(e)
    return tube
    
    
def inclusion(polytope, constraint_box):
    dimension = len(polytope.vertices()[0])
    for vertice in polytope.vertices(): # 2^3
        for dim in range(dimension):    # 10
            if not constraint_box[dim][0] <= vertice[dim] <= constraint_box[dim][1]:
                return False
    return True


def tube_validation(tube, s, u, constraints_boxes, func):
    N, i = len(tube), 0
    while True:
        if not inclusion(vector(list(s[i])) + tube[i], constraints_boxes[i]):
            break
        if i == N:
            break
        i += 1
    return i    
    # Return index where constraints are not satisfied, if N: all satisfied
