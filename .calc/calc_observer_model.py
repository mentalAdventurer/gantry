import numpy as np
import control as ctrl

L1=1.0
L2=1.0
L=np.sqrt(L1**2+L2**2)/2
M=10
J=M/12*(L1**2+L2**2)
m=0.8
k=100.0

x1=5        # -> Rang der Beobachtbarkeitsmatrix bleibt bei versch. 
x2=10.33    # Ruhelagen gleich.

mass_matrix_lin= np.array([[m+M, 0, -m*x2],
            [0, m+M, m*x1],
            [-m*x2, m*x1, J+m*(x1**2+x2**2)]])

mass_matrix_lin_inv = np.linalg.inv(mass_matrix_lin)

state_vector = -1*np.dot(mass_matrix_lin_inv, np.array([[m, 0],[0, m],[-m*x2, m*x1]]))

A=np.array([[0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0],
                    [0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0],
                    [0 , 0, 0, 0, 0, state_vector[0,0], state_vector[0,1], mass_matrix_lin_inv[0,0], mass_matrix_lin_inv[0,1], mass_matrix_lin_inv[0,2]],
                    [0 , 0, 0, 0, 0, state_vector[1,0], state_vector[1,1], mass_matrix_lin_inv[1,0], mass_matrix_lin_inv[1,1], mass_matrix_lin_inv[1,2]],
                    [0 , 0, 0, 0, 0, state_vector[2,0], state_vector[2,1], mass_matrix_lin_inv[2,0], mass_matrix_lin_inv[2,1], mass_matrix_lin_inv[2,2]],
                    [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                    [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0],
                    [0 , 0, -4*k, 0, 0, 0, 0, 0, 0, 0],
                    [0 , 0, 0, -4*k, 0, 0, 0, 0, 0, 0],
                    [0 , 0, 0, 0, -4*k*L**2, 0, 0, 0, 0, 0],])
       
B=np.array([[0 , 0],[0 , 0],[0 , 0],[0 , 0],[0 , 0],[1 , 0],[0 , 1],[0 , 0],[0 , 0],[0 , 0]])

C=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0]])

O = ctrl.obsv(A, C)

print(O)
print(O.shape)
print(np.linalg.matrix_rank(O)) # -> Rang der Beobachtbarkeitsmatrix ist kleiner als die Zahl der Zustände n -> System ist nicht vollständig Rekonstruierbar

print(A)