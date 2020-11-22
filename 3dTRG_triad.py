# Tensor formulation of some 3d model using triad method 
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
beta = float(1.0/Temp)
Niter=2
Dcut=32   # Result unchanged D=32 onwards! Reasons unknown. 


def dagger(a):

    return np.transpose(a).conj()


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


def Z3d_Ising():

    a = np.sqrt(np.cosh(beta))
    b = np.sqrt(np.sinh(beta)) 
    W = np.array([[a,b],[a,-b]])
    out = np.einsum("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W)
    return out



def coarse_graining(in1, in2, in3, in4,impure=False):


    A = in1
    B = in2
    C = in3
    D = in4 

    T = np.einsum("ika, amb, bnc, clj -> ijklmn", A, B, C, D)

    
    #T = T.transpose(1,2,4,5,0,3)
    #print ("Start shape", np.shape(T))
    # Eq (3) of arXiv:1912.02414

    M = ncon([T,T],[[-1,-3,-5,-7,1,-9],[-2,-4,-6,-8,-10,1]])
    a = np.shape(M)[0]*np.shape(M)[1] 
    b = np.shape(M)[2]*np.shape(M)[3] 
    e = np.shape(M)[4]*np.shape(M)[5] 
    f = np.shape(M)[6]*np.shape(M)[7] 
    c = np.shape(M)[8]
    d = np.shape(M)[9]
    Mprime = np.reshape(M1,(a,(b*e*f*c*d)))
    K = np.dot(Mprime, dagger(Mprime))

    az = int(np.sqrt(a))
    K2 = np.reshape(K,(az,az,az,az))

    U, s1, UL = tensorsvd(K2,[0,1],[2,3],Dcut)  # This is U [Eq. '11' of arXiv:1912.02414] 

    Mprime = np.reshape(M1,(e,a*b*f*c*d))
    K = np.dot(Mprime, dagger(Mprime))

    az = int(np.sqrt(e)) 
    K3 = np.reshape(K,(az,az,az,az))

    V, s2, VL = tensorsvd(K3,[0,1],[2,3],Dcut)  # This is V of arXiv:1912.02414 


    #print ("Shapess", np.shape(C), np.shape(D), np.shape(U), np.shape(D), np.shape(V))
    UC = np.einsum("azc, cqp, pix, bji, qjy -> abyzx", C, D, U, D, V)   
    MC = np.einsum("awc, bwd -> abcd", B, C)
    DC = np.einsum("dzb, pix, pqa, qjy, ijd -> zxyab", B, np.conjugate(U), A, np.conjugate(V), A)


    Tdirect = np.einsum("zxyae, aebf, bfijk -> zjxkyi", DC, MC, UC)


    Tmp = np.einsum("abcd, cdyxz -> abyxz", MC, UC)
    G, st, D = tensorsvd(Tmp,[0,1,2],[3,4],Dcut)
    G = np.einsum("abcd, de -> abce", G, st)

    Tmp2 = np.einsum("zxyab, abig -> zxyig", DC, G) # [Eq. '21' of arXiv:1912.02414] 
    A, st2, MCprime = tensorsvd(Tmp2,[0,1],[2,3,4],Dcut) # [Eq. '21' of arXiv:1912.02414] 
    MCprime = np.einsum("dc, cabe -> dabe", st2, MCprime)  
    B, st3, C = tensorsvd(MCprime,[0,1],[2,3],Dcut)

    B = np.einsum("dce, eb -> dcb", B, st3)

    Tindirect = np.einsum("ika, amb, bnc, clj -> ijklmn", A, B, C, D)

    print ("Norm difference!!", LA.norm(Tindirect) - LA.norm(Tdirect))
    # This vanishes as it should. Indirect is cheaper! 

    return A,B,C,D 



if __name__ == "__main__":
    

    a = np.sqrt(np.cosh(beta))
    b = np.sqrt(np.sinh(beta)) 
    W = np.array([[a,b],[a,-b]])
    Id = np.eye(2) 

    A = np.einsum("ax, ay -> xya", W, W)
    B = np.einsum("ab, az -> azb", Id, W)
    C = np.einsum("bc, bz -> bzc", Id, W)
    D = np.einsum("cy, cx -> cyx", W, W)


    for iter in range (Niter):

        A, B, C, D = coarse_graining(A,B,C,D) 
        A, B, C, D = coarse_graining(A,B,C,D) 


    # ........................................................

    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


            
