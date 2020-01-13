# Tensor formulation of some 3d model
# Started: Dec 5, 2019
# Free energy at T= 4.5115 is -3.51
# Ref: https://arxiv.org/abs/1912.02414

import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
from scipy import special
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix
from sklearn.utils.extmath import randomized_svd
import time
import datetime 
from ncon import ncon

startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 

if len(sys.argv) < 2:
  print("Usage:", str(sys.argv[0]), "<Temperature, T>")
  sys.exit(1)

Temp =  float(sys.argv[1])
#h =  float(sys.argv[2])
beta = float(1.0/Temp)
BETAAA = float(sys.argv[1])
Niter=4
D=5


def tensorsvd(input,left,right,D):
    '''Reshape an input tensor into a rectangular matrix with first index corresponding
    to left set of indices and second index corresponding to right set of indices. Do SVD
    and then reshape U and V to tensors [left,D] x [D,right]  
    '''

    T = np.transpose(input,left+right)
    left_index_list = []
    for i in range(len(left)):
        left_index_list.append(T.shape[i])
    xsize = np.prod(left_index_list) 
    right_index_list = []
    for i in range(len(left),len(left)+len(right)):
        right_index_list.append(T.shape[i])
    ysize = np.prod(right_index_list)
    T = np.reshape(T,(xsize,ysize))
    
    
    U, s, V = np.linalg.svd(T,full_matrices = False)
    #U, s, V = randomized_svd(T, n_components=D, n_iter=4,random_state=None)
    
    if D < len(s):
        s = np.diag(s[:D])
        U = U[:,:D]
        V = V[:D,:]
    else:
        D = len(s)
        s = np.diag(s)

    U = np.reshape(U,left_index_list+[D])
    V = np.reshape(V,[D]+right_index_list)
        
        
    return U, s, V


def Z3d_Ising():

    a = np.sqrt(np.cosh(beta))
    b = np.sqrt(np.sinh(beta)) 
    W = np.array([[a,b],[a,-b]])
    out = np.einsum("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W)
    return out


def Z3d_U1():

    A = np.zeros([7, 7]) 

    for i in range (7):
        for j in range (7):
            A[i][j] = sp.special.iv(i-j, BETAAA)

    L = LA.cholesky(A) 
    out = np.einsum("ia, ib, ic, id, ie, if -> abcdef", L, L, L, L, L, L)
    return out



def coarse_graining(input,impure=False):
    
    # Z-direction

    
    #A = ncon([input,input],[[-1,-3,-5,-7,1,-10],[-2,-4,-6,-8,-9,1]])
    #Ux, s, V = tensorsvd(A,[2,3],[0,1,4,5,6,7,8,9],D)
    #Uy, s, V = tensorsvd(A,[0,1],[2,3,4,5,6,7,8,9],D)
    #A = ncon([Ux,Uy,A,Uy,Ux],[[3,4,-2],[1,2,-1],[1,2,3,4,5,6,7,8,-5,-6],[5,6,-3],[7,8,-4]])
    

    
    #A = ncon([input,input,input,input],[[-1,5,-5,2,1,3],[-2,6,-6,8,4,1],[-3,5,-7,2,7,3],[-4,6,-8,8,4,7]])

    A = ncon([input,input,input,input],[[-1,5,-5,2,1,3],[-2,6,-6,2,4,1],[-3,5,-7,8,7,3],[-4,6,-8,8,4,7]])
    Ux, s, V = tensorsvd(A,[2,3],[0,1,4,5,6,7],D)
    Uy, s, V = tensorsvd(A,[0,1],[2,3,4,5,6,7],D)
    A = ncon([Ux,Uy,input,input,Uy,Ux],[[3,4,-2],[1,2,-1],[1,3,5,7,10,-6],[2,4,6,8,-5,10],[5,6,-3],[7,8,-4]])

    if impure:
        B = ncon([b,input],[[-1,-3,-5,-7,1,-10],[-2,-4,-6,-8,-9,1]])    
        B = ncon([Ux,Uy,B,Uy,Ux],[[3,4,-2],[1,2,-1],[1,2,3,4,5,6,7,8,-5,-6],[5,6,-3],[7,8,-4]])

    # Y-direction
    AA = ncon([A,A],[[-1,-2,1,-6,-7,-8],[1,-3,-4,-5,-9,-10]])
    Uz, s, V  = tensorsvd(AA,[6,8],[0,1,2,3,4,5,7,9],D)
    Ux, s, V  = tensorsvd(AA,[1,2],[0,3,4,5,6,7,8,9],D) 
    AA = ncon([Uz,Ux,AA,Ux,Uz],[[5,7,-5],[1,2,-2],[-1,1,2,-3,3,4,5,6,7,8],[4,3,-4],[6,8,-6]])  
    if impure:
        BA = ncon([B,A],[[-1,-2,1,-6,-7,-8],[1,-3,-4,-5,-9,-10]])
        BA = ncon([Uz,Ux,BA,Ux,Uz],[[5,7,-5],[1,2,-2],[-1,1,2,-3,3,4,5,6,7,8],[4,3,-4],[6,8,-6]])
    
    # X-direction
    AAAA = ncon([AA,AA],[[-2,-3,-4,1,-7,-8],[-1,1,-5,-6,-9,-10]])
    Uz, s, V  = tensorsvd(AAAA,[6,8],[0,1,2,3,4,5,7,9],D)
    Uy, s, V  = tensorsvd(AAAA,[3,4],[0,1,2,5,6,7,8,9],D)
    AAAA = ncon([Uz,Uy,AAAA,Uy,Uz],[[5,7,-5],[3,4,-3],[1,2,-2,3,4,-4,5,6,7,8],[2,1,-1],[6,8,-6]])
    if impure:
        BAAA = ncon([BA,AA],[[-2,-3,-4,1,-7,-8],[-1,1,-5,-6,-9,-10]])
        BAAA = ncon([Uz,Uy,BAAA,Uy,Uz],[[5,7,-5],[3,4,-3],[1,2,-2,3,4,-4,5,6,7,8],[2,1,-1],[6,8,-6]])
    
    maxAAAA = np.max(AAAA)

    AAAA = AAAA/maxAAAA # Normalize by largest element of the tensor
    if impure:
        BAAA = BAAA/maxAAAA
    else:
        BAAA = AAAA
    
        
    return AAAA, BAAA, maxAAAA


def final_step(input,impure=False):


    M1 = ncon([input,input,input,input],[[1,2,3,4,-1,-2],[3,5,1,6,-3,-4],[8,4,7,2,-5,-6],[7,6,8,5,-7,-8]])
    Z = ncon([M1, M1],[[1,2,3,4,5,6,7,8], [2,1,4,3,6,5,8,7]])

    return Z 


if __name__ == "__main__":
    



    T = Z3d_Ising()   # Get the initial tensor 
    #T = Z3d_U1() 
    norm = np.max(T)
    T /= norm
    M1 = ncon([T,T,T,T],[[1,2,3,4,-1,-2],[3,5,1,6,-3,-4],[8,4,7,2,-5,-6],[7,6,8,5,-7,-8]])
    Z = ncon([M1, M1],[[1,2,3,4,5,6,7,8], [2,1,4,3,6,5,8,7]])
    N = 1
    C = np.log(norm)
    #f = -Temp*(np.log(Z)+6*C)/(6)
    f = -(np.log(Z)+6*C)/(6)


    for i in range(Niter):

        print ("Coarse graining, step ", i+1)
        T, TI, norm = coarse_graining(T, False)
        C = np.log(norm)+6*C 
        N *= 6
        f = -Temp*(np.log(Z)+6*C)/(6*N)
        #f = -(np.log(Z)+6*C)/(6*N)
        print ("Free energy ", f)


        if i == Niter-1:

            Z = final_step(T, False)
            f = -Temp*(np.log(Z)+6*C)/(6*N)
            #f = -(np.log(Z)+6*C)/(6*N)


    print ("Free energy is ", f, " at T= ", Temp)
    #print ("Free energy is ", f*BETAAA, " at beta= ", BETAAA)
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


            

        
