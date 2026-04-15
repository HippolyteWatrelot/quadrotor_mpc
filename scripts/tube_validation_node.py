import rospy
import numpy as np
from quadrotor_mpc.optim_utils import *
from sage.all import Polyhedron

from quadrotor_mpc.srv import Tube

nx, nu = 10, 4
D, W, state_noise = None, None, None
Q, R = None, None
states_keys, inputs_keys = None, None
d_config = None
c_func = None


def get_DW():
    global D, W, state_noise, dt, mass
    D = np.zeros([nx, 3])
    for i in range(3):
        D[i, i] = dt / mass
    W_vertices = rospy.get_param('W_vertices')
    W = Polyhedron(vertices=[vertice for vertice in W_vertices])
    state_noise = W.linear_transform(D)
    
def get_QR():
    global Q, R
    Q = np.diag(rospy.get_param('Q'))
    R = np.diag(rospy.get_param('R'))
    assert Q.shape[0], R.shape[0] == nx, nu
    
def get_keys():
    global states_keys, inputs_keys
    states_keys = rospy.get_param('states_keys')
    inputs_keys = rospy.get_params('inputs_keys')



def tube_process(req):

    global D, W, Q, R, state_noise, states_keys, inputs_keys, c_func
    
    states, inputs, u_nom0, w0, current_e, x, N = req.pred_states, req.pred_inputs, np.array(u_nom0), np.array(req.w0), np.array(req.current_e), np.array(req.x), req.N    
    # x: current state, u_nom0: predicted nominal first input, w0: current noise, current_e: current perturbation part state, N: horizon length
    constraints_boxes = req.constraint_boxes
    constraint_boxes = constraints_boxes.reshape([N, int(constraint_boxes.shape[0]/(N*2)), 2])               # (N, nx+nu, 2)
    
    A_list, B_list = states_sequence_linearization(states, inputs, state_keys, input_keys, d_config, dt)
    K_N = discrete_Riccati(A_list[-1], B_list[-1], Q, R)
    AB_list = np.concatenate((A_list, B_list), axis=1)
    K_list = LTV_LQR(AB_list, K_N, P, Q, R, N)    # Back recursive algorithm
    new_e = (A_list[0] + B_list[0] @ K_list[0]) @ init_e + D @ w0
    u0 = u_nom0 + K_list[0] @ (x - current_e)
    
    tube = variable_tube(AB_list, K_list, state_noise, init_e)                  # state-noise tube obtained with linearized system on each horizon point
    n_validation = tube_validation(tube, states, inputs, constraints_boxes, c_func)

    return u0, new_e, n_validation



def tube_validation_node():
    rospy.init_node("tube_validation_node")
    s_traj = rospy.Service("tube_validation", Tube, tube_process)
    rospy.spin()


if __name__ == "__main__":
    der_file = "/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/yaml/Linear_Coefficients/reduced_partial_derivatives_values.yaml"
    c_func = ca.Function.load('/home/hippolytewatrelot/drone_ws/src/quadrotor_mpc/scripts/CasADi_formulas/discrete_dyn_f.casadi')
    with open(der_file, "r") as f1:
        d_config = yaml.safe_load(f1)
    mass = rospy.get_param('mass')
    D, W = get_DW()
    Q, R = get_QR()
    states_keys, inputs_keys = get_keys()
    tube_validation_node()
