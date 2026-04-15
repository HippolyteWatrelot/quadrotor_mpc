import numpy as np
import scipy
import yaml
import os

from numpy.linalg import matrix_power

A = np.zeros([12, 12])
B = np.zeros([12, 8])

def make_control_matrix(A, B, N):
    dim0 = B.shape[0]
    dim1 = B.shape[1]
    C = np.zeros([N*dim0, N*dim1])
    C[B.shape[0]:2*B.shape[0], 0:B.shape[1]] = B
    for i in range(2, N):
        for j in range(i-1):
            C[i*B.shape[0]:(i+1)*B.shape[0], j*B.shape[1]:(j+1)*B.shape[1]] = matrix_power(A, (i-1-j)) @ B
        C[i*B.shape[0]:(i+1)*B.shape[0], (i-1)*B.shape[1]:i*B.shape[1]] = B
    return C
    
def S(A, B, N):
    dim0 = B.shape[0]
    dim1 = B.shape[1]
    C = np.zeros([N*dim0, N*dim1])
    C[0:dim0, 0:dim1] = B
    for i in range(1, N):
        for j in range(i):
            C[i*dim0:(i+1)*dim0, j*dim1:(j+1)*dim1] = matrix_power(A, (i-j)) @ B
        C[i*dim0:(i+1)*dim0, i*dim1:(i+1)*dim1] = B
    return C
    
def Phi(A, N):
    C = np.zeros([A.shape[0]*N, A.shape[1]])
    C[:A.shape[0], :A.shape[1]] = A
    for i in range(1, N):
        C[i*A.shape[0]:(i+1)*A.shape[0], :A.shape[1]] = matrix_power(A, i+1)
    return C
    
def make_Ap(A):
    Ap = np.zeros([A.shape[0], N*A.shape[1]])
    Ap[:, :A.shape[1]] = np.eye(A.shape[1])
    mat = A
    for i in range(1, N):
        Ap[:, i*A.shape[1]:(i+1)*A.shape[1]] = mat
        mat = A @ mat;
    return Ap
    
def make_K(A, B, N, C):
    C = make_control_matrix(A, B, N)
    Ap = make_Ap(A)
    F = (Ap @ Q @ C).T
    H = R + C.T @ Q @ C
    K = -np.linalg.inv(H) @ F
    return K[:B.shape[1]]
    
def make_Mx(A, N, K):
    Mx = np.zeros([N * A.shape[0], A.shape[1]])
    Mx[:A.shape[0], :A.shape[1]] = np.eye(A.shape[0])
    M = A + B @ K
    T = A + B @ K
    for i in range(1, N + 1):
        Mx[(i - 1) * A.shape[0]:i * A.shape[0], :A.shape[1]] = M
        M = T @ M
    return Mx

def make_Mc(A, B, K, N):
    Mc = np.zeros([N*B.shape[0], N*B.shape[1]])
    Mc[:B.shape[0], :B.shape[1]] = B
    phi = A + B @ K
    for i in range(1, N):
        for j in range(i):
            Mc[i*B.shape[0]:(i+1)*B.shape[0], j*B.shape[1]:(j+1)*B.shape[1]] = matrix_power(phi, i-j) @ B
        Mc[i * B.shape[0]:(i + 1) * B.shape[0], i * B.shape[1]:(i + 1) * B.shape[1]] = B
    return Mc
    
def Riccati(A, B, Q, R):
    P = Q
    while True:
        P_new = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if np.linalg.norm(P_new - P) < 1e-9:
            P = P_new
            break
        P = P_new
    return -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A            # K
    
    
if __name__ == "__main__":

    path = os.path.abspath(os.getcwd())
    with open("yaml/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    N = config['N']
    Q = np.diag(config['Q'])
    R = np.diag(config['R'])
    NQ = np.diag(N*config['Q'])
    NR = np.diag(N*config['R'])
    dt = config['delta_t']
    gamma_x = config['kpx'] + config['kix']*dt + config['kdx']/dt
    gamma_y = config['kpy'] + config['kiy']*dt + config['kdy']/dt
    gamma_z = config['kpz'] + config['kiz']*dt + config['kdz']/dt
    gamma_yaw = config['kpyaw'] + config['kiyaw']*dt + config['kdyaw']/dt
    rx, ry, rz, ryaw = dt / (dt + config['taux']), dt / (dt + config['tauy']), dt / (dt + config['tauz']), dt / (dt + config['tauyaw'])
    A[:3, :3] = np.array([[1, dt, dt**2/2], 
                          [0, 1, dt], 
                          [0, -gamma_x, 0]])
    A[3:6, 3:6] = np.array([[1, dt, dt**2/2], 
                            [0, 1, dt], 
                            [0, -gamma_y, 0]])
    A[6:9, 6:9] = np.array([[1, dt, dt**2/2], 
                            [0, 1, dt], 
                            [0, -gamma_z, 0]])
    A[9:12, 9:12] = np.array([[1, dt, dt**2/2], 
                              [0, 1, dt], 
                              [0, -gamma_yaw, 0]])
    B[:3, :2] = np.array([[0, 0], 
                          [0, 0], 
                          [(1-rx)*gamma_x, rx*gamma_x]])
    B[3:6, 2:4] = np.array([[0, 0], 
                            [0, 0], 
                            [(1-ry)*gamma_y, ry*gamma_y]])
    B[6:9, 4:6] = np.array([[0, 0], 
                            [0, 0], 
                            [(1-rz)*gamma_z, rz*gamma_z]])
    B[9:12, 6:8] = np.array([[0, 0], 
                             [0, 0], 
                             [(1-ryaw)*gamma_yaw, rx*gamma_yaw]])
    K = Riccati(A, B, Q, R).flatten()
    #C = make_control_matrix(A, B, N)
    #print(Ap.shape, Q.shape, C.shape)
    #K = make_K(A, B, N, C)
    #Mx = make_Mx(A, N, K)
    #Mc = make_Mc(A, B, K, N)
    #data = {'K': K[:B.shape[0], :].tolist(), 'NQ': Q.tolist(), 'NR': R.tolist(), 'Mx': Mx.tolist(), 'Mc': Mc.tolist()}
    S = S(A, B, N)
    Sf = S.flatten()
    phi = Phi(A, N)
    phif = phi.flatten()
    H = S.T @ NQ @ S + NR
    Hf = H.flatten()
    data = {'K': K.tolist(), 'H': Hf.tolist(), 'Gamma': Sf.tolist(), 'phi': phif.tolist()}
    with open('yaml/mat_utils.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=None)

