# Tensor formulation of some 3d model
# This seems to work after changes on Jul 31, 2021
# Not in use 
# Tests done on 3d Ising cubic lattice by comparing with triads

import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
from scipy import special
from scipy.linalg import sqrtm
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix
from sklearn.utils.extmath import randomized_svd
import time
import datetime 
#from ncon import ncon
from opt_einsum import contract

startTime = time.time()
print ("STARTED:" , datetime.datetime.now().strftime("%d %B %Y %H:%M:%S")) 

if len(sys.argv) < 4:
  print("Usage:", str(sys.argv[0]), "<Temperature, Niter, Dcut>")
  sys.exit(1)

Temp =  float(sys.argv[1])
beta = float(1.0/Temp)
Niter = int(sys.argv[2])
D = int(sys.argv[3])


def tensorsvd(input,left,right,D):

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
    out = contract("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W)
    return out

def CG_direction(input):

    A = contract("abcdef,ghijfl->agbhcidjel", input, input)
    Ux, s, V = tensorsvd(A,[0,1],[2,3,4,5,6,7,8,9],D)
    Uy, s, V = tensorsvd(A,[4,5],[0,1,2,3,6,7,8,9],D)
    #output = ncon([Ux,Ux,A,Uy,Uy],[[3,4,-2],[1,2,-1],[1,2,3,4,5,6,7,8,-5,-6],[5,6,-3],[7,8,-4]])
    output = contract("abc,def,deabpqrstw,pqj,rsz->fcjztw", Ux, Ux, A, Uy, Uy)
    out = np.transpose(output,(4,5,0,1,2,3))
    # We do three rotations: xx'yy'zz' -> zz'xx'yy' -> yy'zz'xx' -> xx'yy'zz'; 

    return out

def coarse_graining(input):
    
    A = CG_direction(input) 
    AA = CG_direction(A) 
    AAAA = CG_direction(AA)
    maxAAAA = np.max(AAAA)
    AAAA = AAAA/maxAAAA 
    # Normalize by largest element of the tensor

    return AAAA, maxAAAA


def final_step(input):

    #M1 = ncon([input,input,input,input],[[1,2,3,4,-1,-2],[3,5,1,6,-3,-4],[8,4,7,2,-5,-6],[7,6,8,5,-7,-8]])
    M1 = contract("abcdpq,ceafrs,hdgbtu,gfhevw -> pqrstuvw", input, input, input, input)
    #Z = ncon([M1, M1],[[1,2,3,4,5,6,7,8], [2,1,4,3,6,5,8,7]])
    Z = contract("abcdefgh,badcfehg", M1, M1)
    # Can simplify this probably. TODO 

    return Z 


if __name__ == "__main__":
    
    T = Z3d_Ising()   # Get the initial tensor for 3d Ising 
    N = 1.
    C = 0.

    for i in range(Niter):

        T, norm = coarse_graining(T)
        C = np.log(norm)+8*C 
        N *= 8

        if i == Niter-1:

            Z = final_step(T)
            f = -Temp*(np.log(Z)+8*C)/(8*N)

    print (Temp, f)
    print ("COMPLETED:" , datetime.datetime.now().strftime("%d %B %Y %H:%M:%S")) 
