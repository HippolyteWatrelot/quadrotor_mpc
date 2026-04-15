import numpy as np
import osqp
import rospy
import casadi as ca
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Int16
from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped



delta_t = 0.1
dt = 0.01
kpx, kpy, kpz, kpyaw = 5, 5, 5, 5
kix, kiy, kiz, kiyaw = 1, 1, 1, 2.5
kdx, kdy, kdz, kdyaw = 0, 0, 0, 0
lxy_tau, lz_tau, az_tau = 0.05, 0.05, 0.1
#r = [delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lz_tau), delta_t / (delta_t + az_tau)]
r = None
p = []
gamma_x, gamma_y, gamma_z, gamma_yaw = kpx + kix*delta_t, kpy + kiy*delta_t, kpz + kiz*delta_t, kpyaw + kiyaw*delta_t


#A =    ca.DM([[1, delta_t, (delta_t**2)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 1, delta_t, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, -gamma_x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 1, delta_t, (delta_t**2)/2, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 1, delta_t, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, -gamma_y, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 1, delta_t, (delta_t**2)/2, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 1, delta_t, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, -gamma_z, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, delta_t, (delta_t**2)/2],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, delta_t],
#              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -gamma_yaw, 0]])

#B =    ca.DM([[0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [(1-r[0])*gamma_x, r[0]*gamma_x, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, (1-r[1])*gamma_y, r[1]*gamma_y, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, (1-r[2])*gamma_z, r[2]*gamma_z, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, (1-r[3])*gamma_yaw, r[3]*gamma_yaw]])


A = None
B = None

N = 20
s = 5
nx, nu = 12, 8
nvars = N * (nx + nu)
trajL = 12*N
#Q = np.diag([10, 1, 0.1, 10, 1, 0.1, 10, 1, 0.1, 10, 1, 0.1])
Q = np.diag([10, 1, 0.1, 10, 1, 0.1, 10, 1, 0.1, 10, 1, 0.1])
#P = np.diag([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
#P = np.diag([100, 1, 0.1, 100, 1, 0.1, 100, 1, 0.1, 100, 1, 0.1])
#P = 50*Q
P = Q
#Q = ca.diag([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
R = np.diag([0.001, 0.01, 0.001, 0.01, 0.001, 0.01, 0.001, 0.01])
#R_delta = np.diag([0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1])
#R_delta_diff = np.diag([1, 1, 1, 1])
#NQ = np.diag(N*[10, 3, 1, 10, 3, 1, 10, 3, 1, 10, 3, 1])
#NR = np.diag(N*[0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1])

H = np.zeros([N*nu, N*nu])
Gamma = np.zeros([N*nx, N*nu])
Phi = np.zeros([N*nx, nx])

umin = ca.DM([-2, -2, -2, -2, -2, -2, -2, -2])
umax = ca.DM([2, 2, 2, 2, 2, 2, 2, 2])

current_pos, current_ori = np.zeros(3), np.zeros(4)
current_pos[2] = 0.2
current_vel, current_acc = np.zeros(3), np.zeros(3)
current_yaw, current_yaw_vel, current_yaw_acc = 0, 0, 0

x_ref = np.array(np.zeros([nx*N, 1]))


def build_AB():
    global delta_t, N, r
    global lxy_tau, lz_tau, az_tau
    global kpx, kix, kpy, kiy, kpz, kiz, kpyaw, kiyaw
    global A, B, r
    r = [delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lxy_tau), delta_t / (delta_t + lz_tau), delta_t / (delta_t + az_tau)]
    gamma_x, gamma_y, gamma_z, gamma_yaw = kpx + kix*delta_t, kpy + kiy*delta_t, kpz + kiz*delta_t, kpyaw + kiyaw*delta_t
    A = np.array([[1, delta_t, (delta_t**2)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, delta_t, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, -gamma_x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, delta_t, (delta_t**2)/2, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, delta_t, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -gamma_y, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, delta_t, (delta_t**2)/2, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, delta_t, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, -gamma_z, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, delta_t, (delta_t**2)/2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, delta_t],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -gamma_yaw, 0]])
    B = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [(1-r[0])*gamma_x, r[0]*gamma_x, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, (1-r[1])*gamma_y, r[1]*gamma_y, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, (1-r[2])*gamma_z, r[2]*gamma_z, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, (1-r[3])*gamma_yaw, r[3]*gamma_yaw]])
    


def traj_length_callback(msg):
    global trajL, x_ref
    trajL = msg.data
    #x_ref = np.zeros(nx*trajL)
    x_ref = ca.DM(np.zeros([nx*(trajL+s*N), 1]))
    print("x_ref length: ", trajL)

def traj_points_callback(tab):
    global x_ref, trajL
    assert(x_ref is not None)
    for i in range(trajL):
        x_ref[nx*i] = tab.data[3*i]
        x_ref[nx*i + 3] = tab.data[3*i+1]
        x_ref[nx*i + 6] = tab.data[3*i+2]
    for i in range(trajL, trajL+s*N):
        x_ref[nx*i] = x_ref[nx*(trajL-1)]
        x_ref[nx*i + 3] = x_ref[nx*(trajL-1) + 3]
        x_ref[nx*i + 6] = x_ref[nx*(trajL-1) + 6]
    print("got points")
        
def traj_speeds_callback(tab):
    global x_ref, trajL
    assert(x_ref is not None)
    for i in range(trajL):
        x_ref[nx*i + 1] = tab.data[3*i]
        x_ref[nx*i + 4] = tab.data[3*i+1]
        x_ref[nx*i + 7] = tab.data[3*i+2]
    for i in range(trajL, trajL+s*N):
        x_ref[nx*i + 1] = 0 #x_ref[nx*(trajL-1) + 1]
        x_ref[nx*i + 4] = 0 #x_ref[nx*(trajL-1) + 4]
        x_ref[nx*i + 7] = 0 #x_ref[nx*(trajL-1) + 7]
    print("got speeds")
        
def traj_accs_callback(tab):
    global x_ref, trajL
    assert(x_ref is not None)
    for i in range(trajL):
        x_ref[nx*i + 2] = tab.data[3*i]
        x_ref[nx*i + 5] = tab.data[3*i+1]
        x_ref[nx*i + 8] = tab.data[3*i+2]
    for i in range(trajL, trajL+s*N):
        x_ref[nx*i + 2] = 0 #x_ref[nx*(trajL-1) + 2]
        x_ref[nx*i + 5] = 0 #x_ref[nx*(trajL-1) + 5]
        x_ref[nx*i + 8] = 0 #x_ref[nx*(trajL-1) + 8]
    print("got accs")
        
def traj_yaws_callback(tab):
    global x_ref, trajL
    assert(x_ref is not None)
    for i in range(trajL):
        x_ref[nx*i+9] = tab.data[i]
    for i in range(trajL, trajL+s*N):
        x_ref[nx*i + 9] = x_ref[nx*(trajL-1) + 9]
    print("got yaws")
        
def traj_yaws_speeds_callback(tab):
    global x_ref, trajL
    assert(x_ref is not None)
    for i in range(trajL):
        x_ref[nx*i+10] = tab.data[i]
    for i in range(trajL, trajL+s*N):
        x_ref[nx*i + 10] = 0 #x_ref[nx*(trajL-1) + 10]
    print("got yaws speeds")
   
def traj_yaws_accs_callback(tab):
    global x_ref, trajL
    assert(x_ref is not None)
    for i in range(trajL):
        x_ref[nx*i+11] = tab.data[i]
    for i in range(trajL, trajL+s*N):
        x_ref[nx*i + 11] = 0 #x_ref[nx*(trajL-1) + 11]
    print("got yaws accs")
        
        
        
def get_parameters():
    global delta_t, N
    #Hvec = rospy.get_param("H")
    #Phivec = rospy.get_param("phi")
    #Gammavec = rospy.get_param("Gamma")
    #H = np.array(Hvec).reshape([N*nu, N*nu])
    #Gamma = np.array(Gammavec).reshape([N*nx, N*nu])
    #Phi = np.array(Phivec).reshape([N*nx, nx])
    delta_t = rospy.get_param('delta_t')
    print("delta_t: ", delta_t)
    N = rospy.get_param('Horizon')
    print("Horizon: ", N)
    build_AB()



def test_MPC(fixed_point=False):

    global x_ref, A, B, Q, R, delta_t, r
    print("A:\n", A)
    print("B:\n", B)
    print("Q:\n", Q)
    print("R:\n", R)
    print("delta_t: ", delta_t)
    
    if fixed_point:
        distance = np.sqrt(np.sum(np.square(current_pos - np.array([p[0], p[3], p[6]]))))
        q = int(50*distance)
        Xref = np.array(p*q)
    else:
        Xref = x_ref
        
    x = ca.MX.sym('x', nx)
    u = ca.MX.sym('u', nu)
    f = ca.Function('f', [x, u], [A @ x + B @ u])
    
    u0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    x0 = np.array([current_pos[0], current_vel[0], current_acc[0], current_pos[1], current_vel[1], current_acc[1], current_pos[2], current_vel[2], current_acc[2], current_yaw, current_yaw_vel, current_yaw_acc])
    print("x0: ", x0)
    
    X = ca.MX.sym('X', nx, N+1)
    U = ca.MX.sym('U', nu, N)
    #U_free = ca.MX.sym('U_free', int(nu/2), N)
    #U_constr = ca.MX.sym('U_constr', int(nu/2), N)
    
    cost = 0
    x0_param = ca.MX.sym('x0', nx)
    #x0ref_param = ca.MX.sym('Xref', nx)
    #u0constr_param = ca.MX.sym('u0', int(nu/2))
    u0_param = ca.MX.sym('u0', nu)
    params = ca.vertcat(x0_param, u0_param)
    #params = ca.vertcat(x0_param, u0constr_param)
    #params = ca.vertcat(x0_param, x0ref_param, u0mem_param)
    g = [X[:, 0] - x0_param]
    for t in range(N):
        x_k = X[:, t]
        u_k = U[:, t]
        #u_free_k = U_free[:, t]
        #u_constr_k = U_constr[:, t]
        #u_k = ca.vertcat(ca.vertcat(ca.vertcat(u_constr_k[0], u_free_k[0]), ca.vertcat(u_constr_k[1], u_free_k[1])), ca.vertcat(ca.vertcat(u_constr_k[2], u_free_k[2]), ca.vertcat(u_constr_k[3], u_free_k[3])))
        cost += ca.mtimes([(x_k - Xref[nx*t:nx*(t+1)]).T, Q, x_k - Xref[nx*t:nx*(t+1)]]) + ca.mtimes([u_k.T, R, u_k])
        x_next = f(x_k, u_k)
        g.append(X[:, t+1] - x_next)
    cost += ca.mtimes([(X[:, N] - Xref[nx*N:nx*(N+1)]).T, Q, X[:, N] - Xref[nx*N:nx*(N+1)]])

    for t in range(N-1):
        g.append(U[0, t+1] - r[0]*U[1, t] - (1-r[0])*U[0, t])
        g.append(U[2, t+1] - r[1]*U[3, t] - (1-r[1])*U[2, t])
        g.append(U[4, t+1] - r[2]*U[5, t] - (1-r[2])*U[4, t])
        g.append(U[6, t+1] - r[3]*U[7, t] - (1-r[3])*U[6, t])
        #g.append(U_constr[0, t+1] - r[0]*U_free[0, t] - (1-r[0])*U_constr[0, t])
        #g.append(U_constr[1, t+1] - r[1]*U_free[1, t] - (1-r[1])*U_constr[1, t])
        #g.append(U_constr[2, t+1] - r[2]*U_free[2, t] - (1-r[2])*U_constr[2, t])
        #g.append(U_constr[3, t+1] - r[3]*U_free[3, t] - (1-r[3])*U_constr[3, t])
    g.append(U[0, 0] - u0_param[0])
    g.append(U[2, 0] - u0_param[2])
    g.append(U[4, 0] - u0_param[4])
    g.append(U[6, 0] - u0_param[6])
    #g.append(U_constr[0, 0] - u0constr_param[0])
    #g.append(U_constr[1, 0] - u0constr_param[1])
    #g.append(U_constr[2, 0] - u0constr_param[2])
    #g.append(U_constr[3, 0] - u0constr_param[3])
    lbg = np.array([0] * (nx * (N+1) + 4*(N-1) + 4))
    ubg = np.array([0] * (nx * (N+1) + 4*(N-1) + 4))
    
    xmin = [-5, -3, -3, -5, -3, -3, 0.18, -3, -3, -ca.inf, -ca.pi, -2*ca.pi]
    xmax = [5, 3, 3, 5, 3, 3, 5, 3, 3, ca.inf, ca.pi, 2*ca.pi]
    umin = [-2, -2, -2, -2, -2, -2, -2, -2]
    umax = [2, 2, 2, 2, 2, 2, 2, 2]
    lbx = (N+1)*xmin + N*umin
    ubx = (N+1)*xmax + N*umax
    
    opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    #opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U_free, -1, 1))    
    nlp = {'x': opt_variables, 'f': cost, 'g': ca.vertcat(*g), 'p': params}
    opts = {
             'ipopt.print_level': 0,
             'print_time': 0,
             'ipopt.tol': 1e-6
           }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    X0 = np.tile(x0.reshape(-1, 1), N+1)
    U0 = np.zeros((nu, N))
    #U0_free = np.zeros((int(nu/2), N))
    init_guess = np.vstack([X0.reshape(-1, 1), U0.reshape(-1, 1)])
    #init_guess = np.vstack([X0.reshape(-1, 1), U0_free.reshape(-1, 1)])
    p_val = np.concatenate((x0, u0))
    #p_val = np.concatenate((x0, u0[0:nu:2]))
    #p_val = np.concatenate((x0, Xref[:nx], u0[0:8:2]))
    #p_val = x0

    sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)
        
    s = sol['x']
    print('X')
    for i in range(N+1):
        print(s[nx*i:nx*(i+1)])
    print('\nU')
    for i in range(N):
        print(s[nx*(N+1)+nu*i:nx*(N+1)+nu*(i+1)])
    


def run_MPC(fixed_point=False):

    global x_ref, A, B, Q, R, R_delta, P, delta_t, r, p
    print("A:\n", A)
    print("B:\n", B)
    print("Q:\n", Q)
    print("R:\n", R)
    print("delta_t: ", delta_t)
    
    if fixed_point:
        distance = np.sqrt(np.sum(np.square(current_pos - np.array([p[0], p[3], p[6]]))))
        q = int(50*distance)
        Xref = np.array(p*q)
    else:
        Xref = x_ref
    
    rate = rospy.Rate(1/delta_t)

    x = ca.MX.sym('x', nx)
    u = ca.MX.sym('u', nu)
    f = ca.Function('f', [x, u], [A @ x + B @ u])
    
    u0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    x0 = np.array([current_pos[0], current_vel[0], current_acc[0], current_pos[1], current_vel[1], current_acc[1], current_pos[2], current_vel[2], current_acc[2], current_yaw, current_yaw_vel, current_yaw_acc])
    print("x0: ", x0)
    
    X = ca.MX.sym('X', nx, N+1)
    U = ca.MX.sym('U', nu, N)
    
    cost = 0
    x0_param = ca.MX.sym('x0', nx)
    u0_param = ca.MX.sym('u0', nu)
    params = ca.vertcat(x0_param, u0_param)
    g = [X[:, 0] - x0_param]
    for t in range(N):
        x_k = X[:, t]
        u_k = U[:, t]
        cost += ca.mtimes([(x_k - Xref[nx*t:nx*(t+1)]).T, Q, x_k - Xref[nx*t:nx*(t+1)]]) + ca.mtimes([u_k.T, R, u_k])
        x_next = f(x_k, u_k)
        g.append(X[:, t+1] - x_next)
    cost += ca.mtimes([(X[:, N] - Xref[nx*N:nx*(N+1)]).T, P, X[:, N] - Xref[nx*N:nx*(N+1)]])
    #if not fixed_point:
    #    for k in range(N-1):
    #        cost += ca.mtimes([(U[:, k+1] - U[:, k]).T, R_delta, (U[:, k+1] - U[:, k])])
    #        cost += ca.mtimes([(U[0:nu:2, k] - U[1:nu:2, k]).T, R_delta_diff, (U[0:nu:2, k] - U[1:nu:2, k])])
        
    for t in range(N-1):
        g.append(U[0, t+1] - r[0]*U[1, t] - (1-r[0])*U[0, t])
        g.append(U[2, t+1] - r[1]*U[3, t] - (1-r[1])*U[2, t])
        g.append(U[4, t+1] - r[2]*U[5, t] - (1-r[2])*U[4, t])
        g.append(U[6, t+1] - r[3]*U[7, t] - (1-r[3])*U[6, t])
    g.append(U[0, 0] - u0_param[0])
    g.append(U[2, 0] - u0_param[2])
    g.append(U[4, 0] - u0_param[4])
    g.append(U[6, 0] - u0_param[6])
    
    lbg = np.array([0] * (nx * (N+1) + 4*(N-1) + 4))
    ubg = np.array([0] * (nx * (N+1) + 4*(N-1) + 4))
        
    xmin = [-5, -3, -3, -5, -3, -3, 0.18, -3, -3, -ca.inf, -ca.pi, -2*ca.pi]
    xmax = [5, 3, 3, 5, 3, 3, 5, 3, 3, ca.inf, ca.pi, 2*ca.pi]
    umin = [-2, -2, -2, -2, -2, -2, -2, -2]
    umax = [2, 2, 2, 2, 2, 2, 2, 2]
    lbx = (N+1)*xmin + N*umin
    ubx = (N+1)*xmax + N*umax
    opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
    nlp = {'x': opt_variables, 'f': cost, 'g': ca.vertcat(*g), 'p': params}
    
    opts = {
             'ipopt.print_level': 0,
             'print_time': 0,
             'ipopt.tol': 1e-6
           }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    ind = 0
    pseudo_xsol = np.tile(x0.reshape(-1, 1), N+1)
    u_opt = np.tile(u0.reshape(-1, 1), N)
    #T = rospy.get_time()
    while True:
        X0 = np.tile(x0.reshape(-1, 1), N+1)
        U0 = np.tile(u0.reshape(-1, 1), N)
        #X0 = pseudo_xsol
        #U0 = u_opt
        init_guess = np.vstack([X0.reshape(-1, 1), U0.reshape(-1, 1)])
        p_val = np.concatenate((x0, u0))
        #p_val = np.concatenate((x0, Xref[ind*nx:(ind+1)*nx]))

        sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)
    
        s = sol['x']
        pseudo_xsol = s[:(N+1)*nx].reshape(-1, 1)
        u_opt = s[(N+1)*nx:].reshape(-1, 1)
        uopt = s[(N+1)*nx:(N+1)*nx+nu]
        ind += 1
        
        cmd = Twist()
        cmd.linear.x = np.cos(current_yaw) * uopt[1] + np.sin(current_yaw) * uopt[3]
        cmd.linear.y = -np.sin(current_yaw) * uopt[1] + np.cos(current_yaw) * uopt[3]
        cmd.linear.z = uopt[5]
        cmd.angular.z = uopt[7]
        cmd_vel_pub.publish(cmd)
        rate.sleep()
        rospy.loginfo(f'uopt: {uopt}, time: {rospy.get_time()}')
        rospy.loginfo(f'error: -> x: {Xref[nx*ind]} {current_pos[0]}, y: {Xref[nx*ind+3]} {current_pos[1]}, z: {Xref[nx*ind+6]} {current_pos[2]}, yaw: {Xref[nx*ind+9]} {current_yaw}')
        #while rospy.get_time() < T + ind*delta_t:
        #    pass
        
        #for _ in range(4):
        #    g.pop(-1)
        #g.append(U[0, 0] - r[0]*uopt[1] - (1-r[0])*uopt[0])
        #g.append(U[2, 0] - r[1]*uopt[3] - (1-r[1])*uopt[2])
        #g.append(U[4, 0] - r[2]*uopt[5] - (1-r[2])*uopt[4])
        #g.append(U[6, 0] - r[3]*uopt[7] - (1-r[3])*uopt[6])
        
        cost = 0
        try:
            for t in range(N):
                cost += ca.mtimes([(X[:, t] - Xref[nx*(t+ind):nx*(t+1+ind)]).T, Q, X[:, t] - Xref[nx*(t+ind):nx*(t+1+ind)]]) + ca.mtimes([U[:, t].T, R, U[:, t]])
            cost += ca.mtimes([(X[:, N] - Xref[nx*N:nx*(N+1)]).T, P, X[:, N] - Xref[nx*N:nx*(N+1)]])
            #if not fixed_point:
            #    for k in range(N-1):
            #        cost += ca.mtimes([(U[:, k+1] - U[:, k]).T, R_delta, (U[:, k+1] - U[:, k])])
            #        cost += ca.mtimes([(U[0:nu:2, k] - U[1:nu:2, k]).T, R_delta_diff, (U[0:nu:2, k] - U[1:nu:2, k])])
        except:
            null_cmd = Twist()
            null_cmd.linear.x = 0
            null_cmd.linear.y = 0
            null_cmd.linear.z = 0
            null_cmd.angular.z = 0
            cmd_vel_pub.publish(null_cmd)
            break
        nlp = {'x': opt_variables, 'f': cost, 'g': ca.vertcat(*g), 'p': params}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        x0 = np.array([current_pos[0], current_vel[0], current_acc[0], current_pos[1], current_vel[1], current_acc[1], current_pos[2], current_vel[2], current_acc[2], current_yaw, current_yaw_vel, current_yaw_acc])
        u0 = np.array([float(uopt[i]) for i in range(nu)])



def fix_a_point():
    global p
    p = [float(input("x: ")), 0, 0, float(input("y: ")), 0, 0, float(input("z: ")), 0, 0, float(input("yaw: ")), 0, 0]


def command_callback(msg):
    global current_pos, current_yaw, x_ref
    if msg.data == 13:
        run_MPC(False)
    elif msg.data == 16:
        test_MPC(False)
    elif msg.data == 17:
        fix_a_point()
        run_MPC(True)
    elif msg.data == 18:
        fix_a_point()
        test_MPC(True)
    elif msg.data == 14:
        print(f'\ndrone position:\nx: {current_pos[0]}\ny: {current_pos[1]}\nz: {current_pos[2]}')
        print(f'drone yaw: {current_yaw}')
        print(f'\ndrone speed:\nx: {current_vel[0]}\ny: {current_vel[1]}\nz: {current_vel[2]}')
        print(f'drone yaw speed: {current_yaw_vel}')
        print(f'\ndrone acceleration:\nx: {current_acc[0]}\ny: {current_acc[1]}\nz: {current_acc[2]}')
        print(f'drone yaw acceleration: {current_yaw_acc}')
    elif msg.data == 15:
        for i in range(x_ref.size()[0] // 12):
            print(x_ref[12*i:12*(i+1)])


def pose_callback(pose):
    global current_pos, current_ori, current_vel, current_acc, dt
    prev_pos_x = current_pos[0]
    prev_pos_y = current_pos[1]
    prev_pos_z = current_pos[2]
    current_pos[0] = pose.pose.position.x
    current_pos[1] = pose.pose.position.y
    current_pos[2] = pose.pose.position.z
    current_ori[0] = pose.pose.orientation.w
    current_ori[1] = pose.pose.orientation.x
    current_ori[2] = pose.pose.orientation.y
    current_ori[3] = pose.pose.orientation.z
    prev_vel_x = current_vel[0]
    prev_vel_y = current_vel[1]
    prev_vel_z = current_vel[2]
    current_vel[0] = (current_pos[0] - prev_pos_x) / dt
    current_vel[1] = (current_pos[1] - prev_pos_y) / dt
    current_vel[2] = (current_pos[2] - prev_pos_z) / dt
    current_acc[0] = (current_vel[0] - prev_vel_x) / dt
    current_acc[1] = (current_vel[1] - prev_vel_y) / dt
    current_acc[2] = (current_vel[2] - prev_vel_z) / dt
    
    
def euler_callback(euler):
    global current_yaw, current_yaw_vel, current_yaw_acc
    prev_yaw = current_yaw
    current_yaw = euler.vector.z
    prev_yaw_vel = current_yaw_vel
    current_yaw_vel = (current_yaw - prev_yaw) / dt
    current_yaw_acc = (current_yaw_vel - prev_yaw_vel) / dt
    
    
def cmd_vel_callback(twist):
    global current_command
    current_command[0] = twist.linear.x
    current_command[1] = twist.linear.y
    current_command[2] = twist.linear.z
    current_command[3] = twist.angular.z
    


if __name__ == "__main__":
    rospy.init_node("pympc")
    get_parameters()
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    command_sub = rospy.Subscriber("/command", Int16, command_callback)
    #cmd_vel_sub = rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)
    pose_sub = rospy.Subscriber("/ground_truth_to_tf/pose", PoseStamped, pose_callback)
    euler_sub = rospy.Subscriber("/ground_truth_to_tf/euler", Vector3Stamped, euler_callback)
    traj_length_sub = rospy.Subscriber("/trajectory_length", Int16, traj_length_callback)
    traj_points_sub = rospy.Subscriber("/trajectory_points", Float64MultiArray, traj_points_callback)
    traj_speeds_sub = rospy.Subscriber("/trajectory_speeds", Float64MultiArray, traj_speeds_callback)
    traj_accs_sub = rospy.Subscriber("/trajectory_accs", Float64MultiArray, traj_accs_callback)
    yaws_sub = rospy.Subscriber("/trajectory_yaws", Float64MultiArray, traj_yaws_callback)
    yaws_speeds_sub = rospy.Subscriber("/trajectory_yaws_speeds", Float64MultiArray, traj_yaws_speeds_callback)
    yaws_accs_sub = rospy.Subscriber("/trajectory_yaws_accs", Float64MultiArray, traj_yaws_accs_callback)
    rospy.spin()
