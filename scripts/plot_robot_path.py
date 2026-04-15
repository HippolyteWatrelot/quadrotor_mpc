#! /usr/bin/env python3

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

cf = os.path.abspath(os.getcwd())


if __name__ == '__main__':

    while True:
    
        with open(cf + '/src/mpc_demos/scripts/npy/path.npy', 'rb') as f:
            path = np.array(np.load(f))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        data = np.array(path)
        ax.plot3D(path[:, 0], path[:, 1], path[:, 2], 'g')
            
        plt.show()
        sys.exit()
