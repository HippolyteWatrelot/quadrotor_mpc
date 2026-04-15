#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import rospy
import yaml
import sys
import os

from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from quadrotor_mpc.srv import Equilibrium


all_elements = ['d_xq', 'd_yq', 'd_z', 'roll', 'rolld', 'pitch', 'pitchd', 'yawd']
inputs = ['ux', 'uy', 'uz', 'uwz']
variables = ['d_xq', 'd_yq', 'd_z', 'yawd']
targets = ['target_x', 'target_y', 'target_z', 'target_yaw']
nx, nu = 8, 4
nc = 2*(nx+nu)


# DEFAULT

num_variables = [1, 0, 0, 0]
num_targets = [2, 2, 2, 0]

F = np.array([[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
              [-0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
              [0, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0, -0.1, 0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, -0.01, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0, -0.1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, -0.01, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, -0.01, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
G = np.array([[0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0],
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0.1, 0, 0, 0], 
              [-0.1, 0, 0, 0], 
              [0, 0.1, 0, 0], 
              [0, -0.1, 0, 0], 
              [0, 0, 0.1, 0], 
              [0, 0, -0.1, 0], 
              [0, 0, 0, 0.1], 
              [0, 0, 0, -0.1]])


def Invariant_client(_vars, _targets, F, G):
    rospy.wait_for_service('linear_matrices')
    try:
        Invariant = rospy.ServiceProxy("linear_matrices", Equilibrium)
        Invariant(_targets, _vars, F, G)
        return True
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
        return None
        
        
if __name__ == "__main__":
    try:
        default = bool(int(sys.argv[1]))
    except:
        default = False
    cf = os.path.abspath(os.getcwd())
    if not default:
        num_variables, num_targets = [], []
        F, G = np.zeros([nc, nx]), np.zeros([nc, nu])
        for i, elt in enumerate(variables):
            while True:
                try:
                    num_variables.append(float(input(f'{elt}: ')))
                    break
                except:
                    pass
        for i, elt in enumerate(targets):
            while True:
                try:
                    num_targets.append(float(input(f'{elt}: ')))
                    break
                except:
                    pass
        for i, elt in enumerate(all_elements):
            while True:
                try:
                    F[2*i, :] = np.zeros(i).tolist() + [1/float(input(f'{elt} upper bound: '))] + np.zeros(nx-(i+1)).tolist()
                    F[2*i+1, :] = np.zeros(i).tolist() + [1/float(input(f'{elt} lower bound: '))] + np.zeros(nx-(i+1)).tolist()
                    assert F[2*i, i] > F[2*i+1, i]
                    break
                except:
                    pass
        for i, elt in enumerate(inputs):
            while True:
                try:
                    G[2*nx + 2*i, :] = np.zeros(i).tolist() + [1/float(input(f'{elt} upper bound: '))] + np.zeros(nu-(i+1)).tolist()
                    G[2*nx + 2*i+1, :] = np.zeros(i).tolist() + [1/float(input(f'{elt} lower bound: '))] + np.zeros(nu-(i+1)).tolist()
                    assert G[2*nx+2*i, i] > G[2*nx+2*i+1, i]
                    break
                except:
                    pass
    print('vars: ', num_variables)
    print('targets: ', num_targets)
    print(f'F: {F.shape}, G: {G.shape}')
    Invariant_client(num_variables, num_targets, F.flatten().tolist(), G.flatten().tolist())
