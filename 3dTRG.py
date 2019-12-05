# Tensor formulation of some 3d model
# Started: Dec 5, 2019

import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
from scipy import special
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
import time
import datetime 
from ncon import ncon

if len(sys.argv) < 3:
  print("Usage:", str(sys.argv[0]), "<Temperature, T>  <h> ")
  sys.exit(1)

Temp =  float(sys.argv[1])
h =  float(sys.argv[2])
beta = float(1.0/Temp)
Niter=2


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


def Z3d():
    local_ham = np.array([[-1.0-h,1.0],[1.0,-1.0+h]])
    W = np.exp(-beta*local_ham)
    g = np.zeros((2,2,2,2,2,2))
    g[0,0,0,0,0,0] = 1.
    g[1,1,1,1,1,1] = -1.
    Wsr = LA.sqrtm(W)
    out = ncon([g,Wsr,Wsr,Wsr,Wsr,Wsr,Wsr],[[1,2,3,4,5,6],[-1,1],[-2,2],[3,-3],[4,-4],[-5,5],[6,-6]])
   
    return out



def coarse_graining(input,impure=False):
    
    # Z-direction
    A = ncon([input,input],[[-1,-3,-5,-7,1,-10],[-2,-4,-6,-8,-9,1]])
    Ux, s, V = tensorsvd(A,[2,3],[0,1,4,5,6,7,8,9],D)
    Uy, s, V = tensorsvd(A,[0,1],[2,3,4,5,6,7,8,9],D)
    A = ncon([Ux,Uy,A,Uy,Ux],[[3,4,-2],[1,2,-1],[1,2,3,4,5,6,7,8,-5,-6],[5,6,-3],[7,8,-4]])
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


def final_step(a,impure=False):

	#ap = ncon([a,a,a,a,a,a,a,a] TODO)
    #if impure:
        #bp =  Same as above!


if __name__ == "__main__":


	T = Z3d()   # Get the initial tensor 

	for i in range(Niter):

		if i == Niter-1:
			#M1 = ncon([T,T,T,T],[[1,2,4,5,-1,-2],[4,7,1,8,-3,-4],[24,5,23,2,-5,-6],[23,8,24,7,-7,-8]])
            #M1 = ncon([T,T,T,T],[[11,12,13,14,-1,-2],[13,15,11,16,-3,-4],[20,14,19,12,-5,-6],[19,16,20,15,-7,-8]])
            M1 = ncon([T,T,T,T],[[1,2,3,4,-1,-2],[3,5,1,6,-3,-4],[8,4,7,2,-5,-6],[7,6,8,5,-7,-8]])
            Z = ncon([M1, M1],[[1,2,3,4,5,6,7,8], [2,1,4,3,6,5,8,7]])
            

            # Contract 8 rank-4 tensors to give a number -> is fun!
            # Line 134 & 135 <=> Line 141-142

            Z = ncon([T,T,T,T,T,T,T,T],[[1,2,4,5,6,3],[4,7,1,8,9,10],[11,12,13,14,3,6], \
                [13,15,11,16,10,9],[20,14,19,12,17,18],[19,16,20,15,22,21],[24,5,23,2,18,17],[23,8,24,7,21,22]])

		A, B, C = coarse_graining(T, False)














