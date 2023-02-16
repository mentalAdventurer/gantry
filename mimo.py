import numpy as np
import scipy.linalg as sla
from scipy.special import binom, factorial
import matplotlib.pyplot as plt
from itertools import permutations


# Berechnung aller Kronecker-Indizes zu einem Paar (A,B)
def kronecker(A,B):
    S=controllability_matrix(A,B)
    num_inputs=np.shape(B)[1]
    num_states=np.shape(A)[0]
    ######-------!!!!!!Aufgabe!!!!!!-------------########
    kroneckers=np.ones((num_inputs),dtype=np.int32)
    valid_columns = np.arange(0,num_inputs) 

    i = 1
    while(len(valid_columns) < num_states and i < S.shape[1]):
        columns = np.append(valid_columns,i)
        rank = np.linalg.matrix_rank(S[:,columns])
        if rank == S[:,columns].shape[1]:
            valid_columns = columns
            kroneckers[i%num_inputs]+=1
        i+=1

    ######-------!!!!!!Aufgabe!!!!!!-------------########
    return kroneckers

                      
# Prototyp-Funktion für 5-fach (stückweise)-differenzierbaren  
# Übergang von 0 nach 1 auf dem Intervall [0,1]
def poly_transition(tau,n=0,N=5):
    #tau: array der Zeitpunkte
    #n:   zurückzugebende Ableitung
    
    #maximale Ableitungsordnung ist 5
    assert(n<=N+1)
    
    #vektorielle oder skalare Göße?
    dim = len(np.shape(tau))
    
    
    #sichere, dass tau ein array ist 
    tau = np.atleast_1d(tau)
    
    #lege Rückgabe-Array res als 0-Vektor der gleichen Dimension wie tau an
    res = np.zeros_like(tau)
    
    #ind_m selektiert Zeitpunkte zwischen 0 und 1
    ind_m = np.logical_and(tau >= 0, tau <= 1)

    #Koeffizienten des Übergangspolynoms
    p=factorial(2*N+1)/factorial(N)**2*np.array([binom(N,k)*(-1)**k/(N+k+1) for k in range(N+1)])
    p=np.hstack((p[::-1],np.zeros(N+1,)))
    
    #Auswerten der passenden Ableitung des Polynoms
    res[ind_m]=np.polyval(np.polyder(p,n),tau[ind_m])
    
    #Für 0-te ABleitung muss für Zeiten tau>1 der Rückgabewert auf 1 gesetzt werden
    if(n == 0):
        ind_e = tau > 1
        res[ind_e] = 1.

    if dim==0:
        res=res[0]
    return res

#Steuerbarkeitsmatrix
def controllability_matrix(A,B):
    assert np.ndim(B)<3
    assert np.ndim(A)==2
    
    S=np.zeros((A.shape[0],0))
    Bn=B
    if np.ndim(B)<2:
        Bn=np.atleast2d(B).transpose()

    block=B
    for ii in range(A.shape[0]):
        S = np.hstack((S,block))
        block=A@block
    #print(np.shape(S))
    return S


    
#MIMO-Regelungsform mit gegebenen Steuerbarkeitsmatrizen
def rnf(A, B, C, n):
    assert A.shape[0]==np.sum(n)

    #Steuerbarkeitsmatrix
    S=controllability_matrix(A,B)
    block=B
    for ii in range(A.shape[0]):
        S = np.hstack((S,block))
        block=A@block
    rankS = np.linalg.matrix_rank(S)
    #print(rankS)    
    assert rankS == A.shape[0]
    #print("\n System steuerbar (rank S = {})".format(rankS))

    ninp=np.size(n)

    #Auswahlmatrix
    Sn = np.hstack([S[:,ii:n[ii]*ninp:ninp] for ii in range(ninp)])
    rankSn = np.linalg.matrix_rank(Sn)
    assert rankSn == A.shape[0]

    Sn_inv=np.linalg.inv(Sn)
    
    #Transformationsmatrix
    nu=np.cumsum(n)
    Q=np.zeros((0,A.shape[0]))
    for ii in range(ninp):
        q=Sn_inv[nu[ii]-1,:]
        block=np.vstack([q@np.linalg.matrix_power(A,jj) for jj in range(0,n[ii])])
        Q=np.vstack([Q, block])
    
    #Transformationsmatrix
    Qinv=np.linalg.inv(Q)
    Crnf = C @ Qinv
    Brnf = Q @ B
    Arnf = Q @ A @ Qinv
    #print(Arnf.round(3))
    #print(Brnf.round(3))
    M = np.vstack([Brnf[nu[ii]-1,:] for ii in range(ninp)])
    Bvrnf=Brnf@ np.linalg.inv(M)
    return Arnf, Brnf, Crnf, M, Q, S 

#MIMO-Beobachterform mit gegebenen Steuerbarkeitsmatrizen
def ocf(A, B, C, n):
    Arnf, Brnf, Crnf, M, Q, S = rnf(A.transpose(), C.transpose(), B.transpose(), n)
    return Arnf.transpose(), Crnf.transpose(), Brnf.transpose(), M.transpose(), Q.transpose(), S.transpose()


#MIMO-Ackermannformel
def acker(A, B, eigs):
    """
    Bestimmt Zustandsrückführung u=-Kx mit der Ackermannformel für
    lineare Zweigrößensysteme.
    """
    if (B.ndim==1):
        B=np.atleast_2d(B).transpose()
    m=len(eigs)
    n=[np.size(e) for e in eigs]
    n=np.array(n,dtype='int')
    nu=np.cumsum(n)
    C=np.zeros_like(B).transpose()
    Arnf, Brnf, Crnf, M, Q, S = rnf(A,B,C,n)

    
    q=[Q[sum(n[0:ii])].reshape(1,sum(n)) for ii in range(m)]
    #print(q[0].shape)
    #Koeffizienten des Wunschpolynoms
    coeffs=[np.poly(e)[::-1] for e in eigs]
    K=list()
    #letzter Koeffizient sollte 1 sein
    for ii in range(m):
        #print(ii)
        K.append(q[ii]*coeffs[ii][-1])
        for jj in range(0,n[ii]):
            K[ii]=K[ii]@A+coeffs[ii][n[ii]-jj-1]*q[ii]
    K=(np.vstack(K))
    #print(K.shape)
    K=np.linalg.inv(M) @ K
    K=np.real(K)
    return K
