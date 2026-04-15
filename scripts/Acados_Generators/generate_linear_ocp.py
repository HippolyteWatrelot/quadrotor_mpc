from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver, ACADOS_INFTY
import numpy as np
import rospy
from casadi import MX, SX, vertcat, mtimes, Function, jtimes
import rospkg
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Int16
from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped
import os
import sys
#from my_model import export_model

nx = 12
nu = 8
N = None

dt = None
A, B = None, None
Q, R, P = None, None, None
trajL = 1000                     # Default

current_pos = np.zeros(3)
current_pos[2] = 0.2
current_u = np.zeros(4)
current_yaw = 0
x_max, x_min, u_max, u_min = np.zeros(nx), np.zeros(nx), np.zeros(nu), np.zeros(nu)
r = None

x_ref = np.zeros(nx*trajL)

solvers = ['FULL_CONDENSING_QPOASES', 'PARTIAL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_OSQP']



def traj_length_callback(msg):
    global trajL, x_ref
    trajL = msg.data
    x_ref = np.zeros(nx*(trajL+s*N))
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



def f_mem(symtab, ind):
    global r
    return (1-r[ind]) * symtab[2*ind] + r[ind] * symtab[2*ind+1]


def get_matrices():

    global A, B
    global r, N, dt, Q, R, P, x_ref

    dt = rospy.get_param('delta_t')
    lxy_kp = rospy.get_param("controller/twist/linear/xy/k_p");
    lxy_ki = rospy.get_param("controller/twist/linear/xy/k_i");
    lxy_kd = rospy.get_param("controller/twist/linear/xy/k_d");
    lxy_tau = rospy.get_param("controller/twist/linear/xy/time_constant");
    
    lz_kp = rospy.get_param("controller/twist/linear/z/k_p");
    lz_ki = rospy.get_param("controller/twist/linear/z/k_i");
    lz_kd = rospy.get_param("controller/twist/linear/z/k_d");
    lz_tau = rospy.get_param("controller/twist/linear/z/time_constant");
    
    az_kp = rospy.get_param("controller/twist/angular/z/k_p");
    az_ki = rospy.get_param("controller/twist/angular/z/k_i");
    az_kd = rospy.get_param("controller/twist/angular/z/k_d");
    az_tau = rospy.get_param("controller/twist/angular/z/time_constant");
    
    N = rospy.get_param("Horizon")
    
    r = [dt / (dt + lxy_tau), dt / (dt + lxy_tau), dt / (dt + lz_tau), dt / (dt + az_tau)]
    gamma_x, gamma_y, gamma_z, gamma_yaw = lxy_kp + lxy_ki*dt, lxy_kp + lxy_ki*dt, lz_kp + lz_ki*dt, az_kp + az_ki*dt
    
    A =       np.array([[1, dt, (dt**2)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, -gamma_x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, dt, (dt**2)/2, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -gamma_y, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, dt, (dt**2)/2, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, -gamma_z, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt, (dt**2)/2],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -gamma_yaw, 0]])
                  
    B =       np.array([[0, 0, 0, 0, 0, 0, 0, 0],
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
                 
    Q = np.diag(rospy.get_param("Q_acados"))
    R = np.diag(rospy.get_param("R_acados"))
    P = rospy.get_param("P_factor") * Q
    #Qvec = rospy.get_param("Q")
    #Rvec = rospy.get_param("R")
    #Q = np.diag(Qvec + Rvec[:-1:2])
    #R = np.diag(Rvec[1:-1:2])
    #P = np.block([[rospy.get_param("P_factor") * Q, np.zeros([nx, int(nu/2)])], [np.zeros([int(nu/2), nx]), np.zeros([int(nu/2), int(nu/2)])]])
                  
                  
def get_limits():
    global x_min, x_max, u_max, u_min
    u_min = np.array(rospy.get_param("umin")[1:nu:2])          # 4
    u_max = np.array(rospy.get_param("umax")[1:nu:2])          # 4
    x_min = np.array([-5, -3, -3, -5, -3, -3, 0.18, -3, -3, -ACADOS_INFTY, -np.pi, -2*np.pi, -ACADOS_INFTY, -ACADOS_INFTY, -ACADOS_INFTY, -ACADOS_INFTY])    #16        # includes passive control
    x_max = np.array([5, 3, 3, 5, 3, 3, 5, 3, 3, ACADOS_INFTY, np.pi, 2*np.pi, ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY])                      #16
    
    
    
def pose_callback(msg):
    global current_pos
    current_pos[:] = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    
def euler_callback(msg):
    global current_yaw
    current_yaw = msg.vector.z


def control_callback(msg):
    global current_u
    lx, ly, lz, az = msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z
    try:
        current_u[0] = lx * np.cos(current_yaw) - ly * np.sin(current_yaw)
        current_u[1] = lx * np.sin(current_yaw) + ly * np.cos(current_yaw)
        current_u[2] = lz
        current_u[3] = az
    except: 
        pass
        
    
        
def get_u(z, u):
    global nx, nu
    uf = []
    for i in range(nu // 2):
        uf.append(z[nx + i])
        uf.append(u[i])
    return vertcat(*uf)
    
    
def f_memory(uf):
    global nu, r
    umem = []
    for i in range(nu // 2):
        umem_i = (1 - r[i]) * uf[2*i] + r[i] * uf[2*i + 1]
        umem.append(umem_i)
    return vertcat(*umem)


def generate(qp_solver, c_model=False):
    
    global nx, nu, N, u_min, u_max, x_min, x_max, r
    global x_ref, u_ref
    global A, B, Q, R, P
    
    NX = nx + nu // 2
    NU = nu // 2
    
    model = AcadosModel()
    model.name = "drone_dynamics"
    model.dyn_disc_fun = 'dyn_fun'
    cf = os.path.abspath(os.getcwd())
    print('PATH: ', cf)
    
    p = MX.sym('p', N * (NX+NU))
    
    if not c_model:
        #z = SX.sym('z', NX)  # z
        #u = SX.sym('u', NU)
        px, vx, ax, py, vy, ay, pz, vz, az, yaw, yawd, yawdd, upred1, upred2, upred3, upred4 = MX.sym('px'), MX.sym('vx'), MX.sym('ax'), MX.sym('py'), MX.sym('vy'), MX.sym('ay'), MX.sym('pz'), MX.sym('vz'), MX.sym('az'), MX.sym('yaw'), MX.sym('yawd'), MX.sym('yawdd'), MX.sym('upred1'), MX.sym('upred2'), MX.sym('upred3'), MX.sym('upred4')
        u1, u2, u3, u4 = MX.sym('u1'), MX.sym('u2'), MX.sym('u3'), MX.sym('u4')
        z = vertcat(px, vx, ax, py, vy, ay, pz, vz, az, yaw, yawd, yawdd, upred1, upred2, upred3, upred4)
        u = vertcat(u1, u2, u3, u4)
        x = z[:nx]
        uf = get_u(z, u)
        x_next = mtimes(A, x) + mtimes(B, uf)
        u_mem_next = f_memory(uf)
        z_next = vertcat(x_next, u_mem_next)
        model.x = z
        model.u = u
        model.disc_dyn_expr = z_next
        dyn_fun = Function('f', [z, u], [z_next])
        #model.disc_dyn_fun = dyn_fun
        #model.z = []
        #model.p = p
    else:
        rospy.loginfo('Using .c file')
        model.x = SX.sym('x', NX)
        model.u = SX.sym('u', NU)
        model.z = SX.sym('z', 0)
        model.dyn_ext_fun_type = 'external'
        model.dyn_ext_fun_compile_name = 'drone_dynamics'
        model.dyn_ext_fun_cname = 'dyn_fun'
        model.dyn_disc_fun = 'dyn_fun'
        model.dyn_ext_fun_type = 'generic'
        model.dyn_generic_source = 'drone_dynamics.c'
        #model.dyn_generic_source = 'drone_dynamics_model.c'
        #model.dyn_disc_fun_jac = ''
        #model.dyn_disc_fun_jac_hess = ''
        
    #model.dyn_ext_fun_type = 'discrete'
    
    model.nx = NX
    model.nu = NU
    
    ocp = AcadosOcp()
    ocp.model = model
    
    #rospack = rospkg.RosPack()
    #pkg_path = rospack.get_path('quadrotor_mpc')

    #ocp.dims.N = N
    ocp.solver_options.N_horizon = N
    ocp.dims.nx = NX
    ocp.dims.nu = NU
    ocp.dims.ny = NX + NU
    ocp.dims.ny_e = NX
    
    # Initial conditions
    x_init = np.array([current_pos[0], 0, 0, current_pos[1], 0, 0, current_pos[2], 0, 0, current_yaw, 0, 0, current_u[0], current_u[1], current_u[2], current_u[3]])   # NX = 16
    u_init = np.array([current_u[0], current_u[1], current_u[2], current_u[3]])                                                                                        # NU = 4
    
    print("x_init: ", x_init)
    print("u_init: ", u_init)
    
    # Inequality constraints
    ocp.constraints.x0 = x_init
    #ocp.constraints.x0 = np.concatenate((x_init, u_init), axis=0)
    ocp.constraints.lbu = u_min
    ocp.constraints.ubu = u_max
    ocp.constraints.lbx = x_min
    ocp.constraints.ubx = x_max
    if qp_solver != 'FULL_CONDENSING_QPOASES':
        ocp.constraints.idxbx = np.array(range(NX))
        ocp.constraints.idxbu = np.array(range(NU))
    
    # Equality constraints
    #neq = nu*N
    #p = SX.sym('p', neq)
    #ocp.constraints.expr_h = []
    #ocp.constraints.expr_h += [p[0] - f_mem(u, 0)]
    #ocp.constraints.expr_h += [p[2] - f_mem(u, 1)]
    #ocp.constraints.expr_h += [p[4] - f_mem(u, 2)]
    #ocp.constraints.expr_h += [p[6] - f_mem(u, 3)]
    #for i in range(N-1):
    #    ocp.constraints.expr_h +=     [p[nu*(i+1)] - f_mem(p[nu*i:nu*(i+1)], 0)]
    #    ocp.constraints.expr_h += [p[nu*(i+1) + 2] - f_mem(p[nu*i:nu*(i+1)], 1)]
    #    ocp.constraints.expr_h += [p[nu*(i+1) + 4] - f_mem(p[nu*i:nu*(i+1)], 2)]
    #    ocp.constraints.expr_h += [p[nu*(i+1) + 6] - f_mem(p[nu*i:nu*(i+1)], 3)]
    #ocp.constraints.lh = np.zeros(4*N)
    #ocp.constraints.uh = np.zeros(4*N)
    
    # Cost
    ocp.cost.Vx = np.vstack([np.eye(NX), np.zeros((NU, NX))])  # (20, 16)
    ocp.cost.Vu = np.vstack([np.zeros((NX, NU)), np.eye(NU)])  # (20, 4)
    #ocp.cost.Vx_e = ocp.cost.Vx                                                                      # (ny, nx)
    #ocp.cost.Vx_e[nx:model.nx, nx:model.nx] = np.zeros([model.nu, model.nu])                # Only taking the real state
    ocp.cost.Vx_e = np.eye(NX)
    ocp.cost.W = np.block([
        [Q, np.zeros((NX, NU))],
        [np.zeros((NU, NX)), R]])                                                                    # (ny, ny)
    #ocp.cost.W_e = np.block([[P, np.zeros([nx, NU])], [np.zeros([NU, nx]), np.zeros([NU, NU])]])     # Only taking the real state
    ocp.cost.W_e = P
    ocp.cost.yref_0 = np.array(x_ref[:NX].tolist() + np.zeros(NU).tolist())
    ocp.cost.yref = np.array(x_ref[:NX].tolist() + np.zeros(NU).tolist())                            # (NX + NU,)
    ocp.cost.yref_e = x_ref[:NX]                                                                     # (NX,)
    ocp.cost.cost_type = 'LINEAR_LS'                  # Linear least squares
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    # When Non linear
    #ocp.model.cost_y_expr = vertcat(model.x, model.u)                                                # (NX + NU)
    #ocp.model.cost_y_expr_e = model.x                                                                # (NX,)

    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'DISCRETE'  # 'ERK'
    ocp.solver_options.print_level = 0
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.tf = N * dt                    # (Then deduces dt)
    ocp.solver_options.qp_solver_cond_N = N
    
    print("x: ", ocp.model.x.size())
    print("u: ", ocp.model.u.size())
    print("x_next: ", ocp.model.disc_dyn_expr.size())
    
    print("Vx: ", ocp.cost.Vx.shape)
    print("Vx_e: ", ocp.cost.Vx_e.shape)
    print("Vu: ", ocp.cost.Vu.shape)
    print("W: ", ocp.cost.W.shape)
    print("W_e: ", ocp.cost.W_e.shape)
    print("yref: ", ocp.cost.yref.shape)
    print("yref_e: ", ocp.cost.yref_e.shape)
    
    print("x0: ", ocp.constraints.x0.shape)
    print("lbu: ", ocp.constraints.lbu.shape)
    print("ubu: ", ocp.constraints.ubu.shape)
    print("lbx: ", ocp.constraints.lbx.shape)
    print("ubx: ", ocp.constraints.ubx.shape)
    print("idxbx: ", ocp.constraints.idxbx.shape)
    print("idxbu: ", ocp.constraints.idxbu.shape)
    
    #print("y_expr: ", ocp.model.cost_y_expr.shape)
    #print("y_expr_e: ", ocp.model.cost_y_expr_e.shape)

    ocp.code_export_directory = './c_generated_code'
    ocp_solver = AcadosOcpSolver(ocp, json_file="quadrotor_ocp.json")
    

if __name__ == "__main__":

    c_model = bool(int(sys.argv[1]))
    qp_solver = solvers[int(sys.argv[2])]
    
    rospy.init_node("acados_template")
    
    rospy.loginfo(f'c_model: {c_model}, qp_solver: {qp_solver}')
    
    pose_sub = rospy.Subscriber("/ground_truth_to_tf/pose", PoseStamped, pose_callback)
    euler_sub = rospy.Subscriber("/ground_truth_to_tf/euler", Vector3Stamped, euler_callback)
    control_sub = rospy.Subscriber("/cmd_vel", Twist, control_callback)
    traj_length_sub = rospy.Subscriber("/trajectory_length", Int16, traj_length_callback)
    traj_points_sub = rospy.Subscriber("/trajectory_points", Float64MultiArray, traj_points_callback)
    traj_speeds_sub = rospy.Subscriber("/trajectory_speeds", Float64MultiArray, traj_speeds_callback)
    traj_accs_sub = rospy.Subscriber("/trajectory_accs", Float64MultiArray, traj_accs_callback)
    yaws_sub = rospy.Subscriber("/trajectory_yaws", Float64MultiArray, traj_yaws_callback)
    yaws_speeds_sub = rospy.Subscriber("/trajectory_yaws_speeds", Float64MultiArray, traj_yaws_speeds_callback)
    yaws_accs_sub = rospy.Subscriber("/trajectory_yaws_accs", Float64MultiArray, traj_yaws_accs_callback)

    get_matrices()
    get_limits()
    generate(qp_solver, c_model)
