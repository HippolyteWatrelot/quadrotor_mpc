import numpy as np
import rospy
import rospkg
import os
import sys
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver, ACADOS_INFTY
from casadi import MX, SX, vertcat, mtimes, Function, jtimes
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Int16
from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped
from quadrotor_mpc.transform_utils import euler_to_quaternion, VecToso3, euler_to_rotmatrix, eulerd2w, wd2eulerdd, cross
#from my_model import export_model

nx = 20       # 6 degrees (x, y, z, roll, pitch, yaw) x (p, v, a) + prev_twist_body (x and y)
nu = 10        # twist input on (x, y, z and yaw) + passive ones + passive roll and pitch commands
NX, NU = 26, 4
N = None

dt = None
A, B = None, None
Q, R, P = None, None, None
OIM = None
I = None
g = 9.8065
mass = 1.478
CoG = np.array([-0.000108, 0, -8e-6])
gamma_x, gamma_y, gamma_z, gamma_wx, gamma_wy, gamma_wz = None, None, None, None, None, None
axy_kd = None
trajL = 1000                     # Default

current_pos = np.zeros(3)
current_pos[2] = 0.2
current_u = np.zeros(4)
current_yaw = 0
x_max, x_min, u_max, u_min = np.zeros(nx), np.zeros(nx), np.zeros(nu), np.zeros(nu)
r = None

x_ref = np.zeros(nx*trajL)

solvers = ['FULL_CONDENSING_QPOASES', 'PARTIAL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_OSQP']



def f_mem(symtab, ind):
    global r
    return (1-r[ind]) * symtab[2*ind] + r[ind] * symtab[2*ind+1]


def get_system():

    global A, B
    global r, N, dt, Q, R, P, x_ref
    global gamma_x, gamma_y, gamma_z, gamma_wx, gamma_wy, gamma_wz, axy_kd
    global I, OI

    dt = rospy.get_param('delta_t')
    lxy_kp = rospy.get_param("controller/twist/linear/xy/k_p")
    lxy_ki = rospy.get_param("controller/twist/linear/xy/k_i")
    lxy_kd = rospy.get_param("controller/twist/linear/xy/k_d")
    lxy_tau = rospy.get_param("controller/twist/linear/xy/time_constant")
    
    lz_kp = rospy.get_param("controller/twist/linear/z/k_p")
    lz_ki = rospy.get_param("controller/twist/linear/z/k_i")
    lz_kd = rospy.get_param("controller/twist/linear/z/k_d")
    lz_tau = rospy.get_param("controller/twist/linear/z/time_constant")
    
    axy_kp = rospy.get_param("controller/twist/angular/xy/k_p")
    axy_ki = rospy.get_param("controller/twist/angular/xy/k_i")
    axy_kd = rospy.get_param("controller/twist/angular/xy/k_d")
    
    az_kp = rospy.get_param("controller/twist/angular/z/k_p")
    az_ki = rospy.get_param("controller/twist/angular/z/k_i")
    az_kd = rospy.get_param("controller/twist/angular/z/k_d")
    
    axy_tau = rospy.get_param("controller/twist/angular/xy/time_constant")
    az_tau = rospy.get_param("controller/twist/angular/z/time_constant")
    
    N = rospy.get_param("Horizon")
    
    r = [dt / (dt + lxy_tau), dt / (dt + lxy_tau), dt / (dt + lz_tau), dt / (dt + axy_tau), dt / (dt + axy_tau), dt / (dt + az_tau)]
    gamma_x, gamma_y, gamma_z, gamma_wx, gamma_wy, gamma_wz = [lxy_kp + lxy_ki*dt, lxy_kp + lxy_ki*dt, lz_kp + lz_ki*dt,
                                                               axy_kp + axy_ki*dt, axy_kp + axy_ki*dt, az_kp + az_ki*dt]
    print("r: ", r)
    print("gamma: ", gamma_x, " ", gamma_y, " ", gamma_z, " ", gamma_wx, " ", gamma_wy, " ", gamma_wz)
                 
    Q = np.diag(rospy.get_param("Qnl_acados"))            # real states
    R = np.diag(rospy.get_param("Rnl_acados"))              # active inputs + memory inputs (passive states)
    P = rospy.get_param("P_factor") * Q
    OI = np.array(rospy.get_param("OIM")).reshape([3, 3])
    I = np.array(rospy.get_param("IM")).reshape([3, 3])
    print("Inertia Matrix: ", I)
    #Qvec = rospy.get_param("Q")
    #Rvec = rospy.get_param("R")
    #Q = np.diag(Qvec + Rvec[:-1:2])
    #R = np.diag(Rvec[1:-1:2])
    #P = np.block([[rospy.get_param("P_factor") * Q, np.zeros([nx, int(nu/2)])], [np.zeros([int(nu/2), nx]), np.zeros([int(nu/2), int(nu/2)])]])
                  
                  
def get_limits():
    global x_min, x_max, u_max, u_min
    u_min = np.array(rospy.get_param("umin_nl")[1:nu-2:2])          # 4
    u_max = np.array(rospy.get_param("umax_nl")[1:nu-2:2])          # 4
    x_min = np.array([-5, -3, -3, 
                      -5, -3, -3, 
                      0.18, -3, -3, 
                      -np.pi/2, -10, -10, 
                      -np.pi/2, -10, -10, 
                      -ACADOS_INFTY, -np.pi, -2*np.pi,
                      -ACADOS_INFTY, -ACADOS_INFTY, 
                      -ACADOS_INFTY, -ACADOS_INFTY, -ACADOS_INFTY, -ACADOS_INFTY,
                      -ACADOS_INFTY, -ACADOS_INFTY])                                  #26        # includes passive control
    x_max = np.array([5, 3, 3, 
                      5, 3, 3, 
                      5, 3, 3, 
                      np.pi/2, 10, 10, 
                      np.pi/2, 10, 10, 
                      ACADOS_INFTY, np.pi, 2*np.pi,
                      ACADOS_INFTY, ACADOS_INFTY,
                      ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY,
                      ACADOS_INFTY, ACADOS_INFTY])                                    #26
       
        
    
        
def load_factor(euler):
    q = euler_to_quaternion(euler[2], euler[1], euler[0])
    return 1 / (q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)
    
    
def twist_body_angular(state):
    roll, pitch, yaw = state[9], state[12], state[15]
    rolld, pitchd, yawd = state[10], state[13], state[16]
    wb = eulerd2w([roll, pitch, yaw], [rolld, pitchd, yawd])
    return wb


def space2body(state, vec):
    euler_angles = state[9:-4:3]
    m = euler_to_rotmatrix(euler_angles[2], euler_angles[1], euler_angles[0])
    return np.dot(m.T, vec)


### Building Forces

def get_acceleration_commands(state, command):
    global r, gamma_x, gamma_y, gamma_z
    cart_speeds = state[1:9:3]
    pass_command = state[-4:-1]
    acceleration_command_x = -gamma_x * cart_speeds[0] + (1-r[0])*gamma_x*pass_command[0] + r[0]*gamma_x*command[0]
    acceleration_command_y = -gamma_y * cart_speeds[1] + (1-r[1])*gamma_y*pass_command[1] + r[1]*gamma_y*command[1]
    acceleration_command_z = -gamma_z * cart_speeds[2] + (1-r[2])*gamma_z*pass_command[2] + r[2]*gamma_z*command[2]
    return np.array([acceleration_command_x, acceleration_command_y, acceleration_command_z])


def get_torques(acceleration_command, state, command, twistbody, rel_force):
    global nx, nu, g, I, gamma_wx, gamma_wy, gamma_wz, axy_kd, CoG
    acceleration_command_body = space2body(state, acceleration_command)
    uwx_body, uwy_body = -acceleration_command_body[1]/g, acceleration_command_body[0]/g
    uwx_body_pass, uwy_body_pass = command[8], command[9]
    tbx, tby = twistbody[:2]
    prev_tbx, prev_tby = state[nx-2], state[nx-1]
    uwz_pass, uwz = state[6], command[7]
    wz = eulerd2w([state[9], state[12], state[15]], [state[10], state[13], state[16]])[2]
    lowpass_uwx_body = gamma_wx * ((1-r[3]) * uwx_body_pass + r[3] * uwx_body)
    lowpass_uwy_body = gamma_wy * ((1-r[4]) * uwy_body_pass + r[4] * uwy_body)
    torque_x = I[0, 0] * (gamma_wx * lowpass_uwx_body + axy_kd * ((lowpass_uwx_body - uwx_body_pass)/dt - (tbx - prev_tbx)))
    torque_y = I[1, 1] * (gamma_wy * lowpass_uwy_body + axy_kd * ((lowpass_uwy_body - uwy_body_pass)/dt - (tby - prev_tby)))
    torque_z = I[2, 2] * gamma_wz * (-wz + (1-r[5]) * uwz_pass + r[5] * uwz)
    torques = np.array([torque_x, torque_y, torque_z])
    torques[0] -= CoG[1] * rel_force
    torques[1] -= -CoG[0] * rel_force
    print("torque x: ", torque_x)
    print("torque y: ", torque_y)
    print("torque z: ", torque_z)
    return torques, np.array([lowpass_uwx_body, lowpass_uwy_body])


def get_force(state, acceleration_command_z):
    global g, gamma_z, r
    euler_angles = state[9:-4:3]
    return mass * ((acceleration_command_z - g) * load_factor(euler_angles) + g)


### Forces to Motion

def AdjointTwist(twist):
    w, v = twist[:3], twist[3:]
    return np.block([[VecToso3(w), np.zeros([3, 3])], [VecToso3(v), VecToso3(w)]])
    
def Adjoint(state):
    pos = state[:9:3]
    roll, pitch, yaw = state[9], state[12], state[15] 
    Ad = np.zeros([6, 6])
    RotM = euler_to_rotmatrix(yaw, pitch, roll)
    return np.block([[RotM, np.zeros([3, 3])], [VecToso3(pos) @ RotM, RotM]])
    
def AdjointInvert(state):
    pos = state[:9:3]
    roll, pitch, yaw = state[9], state[12], state[15] 
    AdI = np.zeros([6, 6])
    RotM = euler_to_rotmatrix(yaw, pitch, roll)
    return np.block([[-RotM.T, np.zeros([3, 3])], [-RotM.T @ VecToso3(pos), -RotM.T]])


def get_euler_accs(state, torques):
    global mass, OI, CoG
    euler = np.array([state[9+3*i] for i in range(3)])
    eulerd = np.array([state[10+3*i] for i in range(3)])
    speeds = np.array([state[1+3*i] for i in range(3)])
    body_wrench = np.array(torques.tolist() + [0, 0, 0])
    RotM = euler_to_rotmatrix(euler[2], euler[1], euler[0])
    body_twist = np.array(eulerd2w(euler, eulerd).tolist() + list(RotM.T @ speeds))
    G = np.block([[OI, mass * VecToso3(CoG)], [-mass * VecToso3(CoG), np.diag(mass*np.ones(3))]])
    d_body_twist = np.linalg.inv(G) @ (body_wrench + AdjointTwist(body_twist).T @ G @ body_twist)
    d_wb = d_body_twist[:3]
    euler_accs = wd2eulerdd(euler, eulerd, d_wb)
    return vertcat(euler_accs[0], vertcat(euler_accs[1], euler_accs[2]))
    
    
def get_cartesian_accs(state, rel_force_command):
    global g, mass
    roll, pitch, yaw = state[9], state[12], state[15]
    accs = np.array([rel_force_command/mass * (np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)),
                     rel_force_command/mass * (np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)),
                     rel_force_command/mass * np.cos(pitch)*np.cos(roll) - g])
    return vertcat(accs[0], vertcat(accs[1], accs[2]))


# Passive Inputs

def get_u(z, u):
    # Returns the full command vector
    global nx, nu, NX, NU
    uf = []
    for i in range(NU):
        uf.append(z[nx + i])
        uf.append(u[i])
    uf.append(z[NX-2])
    uf.append(z[NX-1])
    return vertcat(*uf)
    
    
def f_memory(uf, x, passive_internal_commands):
    global nu, NU, r
    umem = []
    for i in range(3):
        umem_i = (1 - r[i]) * uf[2*i] + r[i] * uf[2*i + 1]
        umem.append(umem_i)
    umem_wz = (1 - r[5]) * uf[6] + r[5] * uf[7]
    umem.append(umem_wz)
    umem_wx_body = passive_internal_commands[0]
    umem.append(umem_wx_body)
    umem_wy_body = passive_internal_commands[1]
    umem.append(umem_wy_body)
    return vertcat(*umem)
    
    


def generate(qp_solver, c_model=False):
    
    global nx, nu, N, u_min, u_max, x_min, x_max, r
    global x_ref, u_ref
    global A, B, Q, R, P
    global delta_t
    
    NX = 26   # nx + 6 passive commands
    NU = 4    # nu - 6 passive commands
    
    model = AcadosModel()
    model.name = "full_drone_dynamics"
    model.dyn_disc_fun = 'full_dyn_fun'
    cf = os.path.abspath(os.getcwd())
    print('PATH: ', cf)
    
    p = MX.sym('p', N * (NX+NU))
    
    px, vx, ax, py, vy, ay, pz, vz, az, roll, rolld, rolldd, pitch, pitchd, pitchdd, yaw, yawd, yawdd = MX.sym('px'), MX.sym('vx'), MX.sym('ax'), MX.sym('py'), MX.sym('vy'), MX.sym('ay'), MX.sym('pz'), MX.sym('vz'), MX.sym('az'), MX.sym('roll'), MX.sym('rolld'), MX.sym('rolldd'), MX.sym('pitch'), MX.sym('pitchd'), MX.sym('pitchdd'), MX.sym('yaw'), MX.sym('yawd'), MX.sym('yawdd')
    prev_twist_body_x, prev_twist_body_y = MX.sym('prev_tbx'), MX.sym('prev_tby')
    upass1, upass2, upass3, upass4 = MX.sym('upass1'), MX.sym('upass2'), MX.sym('upass3'), MX.sym('upass4')
    uwx_body_pass, uwy_body_pass = MX.sym('uwx_body_pass'), MX.sym('uwy_body_pass')
    u1, u2, u3, u4 = MX.sym('u1'), MX.sym('u2'), MX.sym('u3'), MX.sym('u4')
    
    z = vertcat(px, vx, ax, py, vy, ay, pz, vz, az, roll, rolld, rolldd, pitch, pitchd, pitchdd, yaw, yawd, yawdd, prev_twist_body_x, prev_twist_body_y, upass1, upass2, upass3, upass4, uwx_body_pass, uwy_body_pass)
    u = vertcat(u1, u2, u3, u4)
    
    x = z[:-6]
    uf = get_u(z, u)
        
    acc_commands = get_acceleration_commands(z, uf)
    tba = twist_body_angular(x)
    relative_z_force = get_force(x, acc_commands[2])
    planar_torques, pic = get_torques(acc_commands, x, uf, tba, relative_z_force)
    output_cart_acc = get_cartesian_accs(x, relative_z_force)
    output_euler_acc = get_euler_accs(x, planar_torques)
    x_next = vertcat(px + dt*vx + (dt**2)*output_cart_acc[0]/2,
                     vx + dt*ax,
                     output_cart_acc[0],
                     py + dt*vy + (dt**2)*output_cart_acc[1]/2,
                     vy + dt*ay,
                     output_cart_acc[1],
                     pz + dt*vz + (dt**2)*output_cart_acc[2]/2,
                     vz + dt*az,
                     output_cart_acc[2],
                     roll + dt*rolld + (dt**2)*output_euler_acc[0]/2,
                     rolld + dt*rolldd,
                     output_euler_acc[0],
                     pitch + dt*pitchd + (dt**2)*output_euler_acc[1]/2,
                     pitchd + dt*pitchdd,
                     output_euler_acc[1],
                     yaw + dt*yawd + (dt**2)*output_euler_acc[2]/2,
                     yawd + dt*yawdd,
                     output_euler_acc[2],
                     tba[0],
                     tba[1])
        
    u_mem_next = f_memory(uf, pic, x)
    z_next = vertcat(x_next, u_mem_next)
    model.x = z
    model.u = u
    model.disc_dyn_expr = z_next
    dyn_fun = Function('f', [z, u], [z_next])
    #model.disc_dyn_fun = dyn_fun
    #model.z = []
    #model.p = p
    
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
    x_init = np.zeros(26)  # NX = 26
    u_init = np.zeros(4)   # NU = 4
    
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
    ocp.cost.Vx = np.vstack([np.eye(NX), np.zeros((NU, NX))])  # (20, 26)
    ocp.cost.Vu = np.vstack([np.zeros((NX, NU)), np.eye(NU)])  # (20, 4)
    #ocp.cost.Vx_e = ocp.cost.Vx                                                                      # (ny, nx)
    #ocp.cost.Vx_e[nx:model.nx, nx:model.nx] = np.zeros([model.nu, model.nu])                # Only taking the real state
    ocp.cost.Vx_e = np.eye(NX)
    ocp.cost.W = np.block([
        [Q, np.zeros((NX, NU))],
        [np.zeros((NU, NX)), R]])                                                                    # (ny, ny)
    #ocp.cost.W_e = np.block([
    #    [P, np.zeros([NX, NU])], 
    #    [np.zeros([NU, NX]), np.zeros([NU, NU])]])                                                   # Only taking the real state
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
    ocp_solver = AcadosOcpSolver(ocp, json_file="full_quadrotor_ocp.json")
    

if __name__ == "__main__":

    c_model = bool(int(sys.argv[1]))
    qp_solver = solvers[int(sys.argv[2])]
    
    rospy.init_node("acados_template")
    
    rospy.loginfo(f'c_model: {c_model}, qp_solver: {qp_solver}')

    get_system()
    get_limits()
    generate(qp_solver, c_model)
