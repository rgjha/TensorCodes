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
Niter=4
Dcut=5


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



#def coarse_graining(input,impure=False):


if __name__ == "__main__":
    

    a = np.sqrt(np.cosh(beta))
    b = np.sqrt(np.sinh(beta)) 
    W = np.array([[a,b],[a,-b]])
    Id = np.eye(2) 

    A = np.einsum("ax, ay -> xya", W, W)
    B = np.einsum("ab, az -> azb", Id, W)
    C = np.einsum("bc, bz -> bzc", Id, W)
    D = np.einsum("cy, cx -> cyx", W, W)



    T = np.einsum("ika, amb, bnc, clj -> ijklmn", A, B, C, D)
    print ("Start shape", np.shape(T))
    # Eq (3) of arXiv:1912.02414


    M = ncon([T,T],[[-1,-3,-5,-7,1,-9],[-2,-4,-6,-8,-10,1]])
    a = np.shape(M)[0] 
    b = np.shape(M)[2]
    e = np.shape(M)[4]
    f = np.shape(M)[6]
    c = np.shape(M)[8]
    d = np.shape(M)[9]
    M1 = np.reshape(M,(a**2,b**2,e**2, f**2, c,d))
    M = M1 
    Mprime = np.reshape(M,(a**2,(b**2)*(e**2)*(f**2)*c*d))
    K = np.matmul(Mprime, dagger(Mprime))

    K = np.reshape(K,(2,2,2,2))
    U1, s1, V1 = tensorsvd(K,[0,1],[2,3],Dcut)  # This is U of arXiv:1912.02414


    Mprime = np.reshape(M,(b**2,(a**2)*(e**2)*(f**2)*c*d))
    K = np.matmul(Mprime, dagger(Mprime))
    K = np.reshape(K, (2,2,2,2))
    U2, s2, V2 = tensorsvd(K,[0,1],[2,3],Dcut)  # This is V of arXiv:1912.02414 

    UC = np.einsum("azc, cqp, pix, bji, qjy -> abyzx", C, D, U1, D, U2)   
    MC = np.einsum("awc, bwd -> abcd", B, C)
    DC = np.einsum("dzb, pix, pqa, qjy, ijd -> zxyab", B, np.conjugate(U1), A, np.conjugate(U2), A)
    T = np.einsum("zxyae, aebf, bfijk -> zjxkyi", DC, MC, UC)
    print ("End shape", np.shape(T))

    Tmp = np.einsum("abcd, cdyxz -> abyxz", MC, UC)
    Gprime, st, Dprime = tensorsvd(Tmp,[0,1,2],[3,4],Dcut)
    Eq20 = np.einsum("abyg, gxz -> abyxz", Gprime, Dprime)
    #print ("Shapes", np.shape(Ut), np.shape(st), np.shape(Vt))


    '''
    for i in range(Niter):

        T, TI, norm = coarse_graining(T, False)
        C = np.log(norm)+6*C 
        N *= 6  # 3d has six open legs 


        if i == Niter-1:

            Z = final_step(T, False)
            f = -Temp*(np.log(Z)+6*C)/(6*N)
            #f = -(np.log(Z)+6*C)/(6*N)
    '''
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


            

        
