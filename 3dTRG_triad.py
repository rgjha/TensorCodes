# Tensor formulation of 3d model using triad method 
# Free energy at T = 4.5115 is -3.51  [is this using Random SVD? .. Probably, yes!]
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
Niter = 15
Dcut = 26


def dagger(a):

    return np.conjugate(np.transpose(a))


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
    # Interchanges between RSVD and usual SVD 
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
    out = ncon((W, W, W, W, W, W),([1,-1],[1,-2],[1,-3],[1,-4],[1,-5],[1,-6]))
    # W_ia * W_ib * W_ic * W_id * W_ie * W_if
    return out


def coarse_graining(in1, in2, in3, in4,impure=False):

    A = in1
    B = in2
    C = in3
    D = in4 

    S1 = ncon((A, np.conjugate(A)),([-1,1,-3], [-2,1,-4]))
    a = np.shape(S1)[0] * np.shape(S1)[1]
    b = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(a,b))

    S2 = ncon((B, np.conjugate(B)),([-1,1,-3], [-2,1,-4]))
    a = np.shape(S2)[0] * np.shape(S2)[1]
    b = np.shape(S2)[2] * np.shape(S2)[3]
    S2 = np.reshape(S2,(a,b))

    Tmp = np.einsum("fyx, iyx -> fi", D, np.conjugate(D))
    R2 = np.einsum("ewf, ijk, fk -> eiwj", C, np.conjugate(C), Tmp)
    a = np.shape(R2)[0] * np.shape(R2)[1]
    b = np.shape(R2)[2] * np.shape(R2)[3]
    R2mat = np.reshape(R2,(a,b))

    Tmp = np.einsum("ijkk -> ij", R2)
    R3 = np.einsum("awb, ijk, bk -> aiwj", B, np.conjugate(B), Tmp)
    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))

    
    tmp1 = np.matmul(S1,S2)
    tmp2 = np.matmul(tmp1,R2mat)
    tmp3 = np.matmul(tmp2,np.transpose(R3mat))
    Kprime = np.matmul(tmp3,np.transpose(S1))
    a = int(np.sqrt(np.shape(Kprime)[0]))
    b = int(np.sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,b,a,a))
    U, s1, UL = tensorsvd(K,[0,1],[2,3],Dcut) 

    # Now finding "V"
    S1 = np.einsum("xyd, xik -> yidk", A, np.conjugate(A))
    a = np.shape(S1)[0] * np.shape(S1)[1]
    b = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(a,b))


    Tmp = np.einsum("ijkk -> ij", R2)
    R3 = np.einsum("awb, ijk, bk -> aiwj", B, np.conjugate(B), Tmp)
    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))

    
    tmp1 = np.matmul(S1,S2)
    tmp2 = np.matmul(tmp1,R2mat)
    tmp3 = np.matmul(tmp2,np.transpose(R3mat))
    Kprime = np.matmul(tmp3,np.transpose(S1))
    
    a = int(np.sqrt(np.shape(Kprime)[0]))
    b = int(np.sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,b,a,a))
    V, s1, VL = tensorsvd(K,[0,1],[2,3],Dcut)
    

    UC = ncon((C, D, U, D, V),([-1,-5,1],[1,2,3],[3,4,-4], [-2,5,4],[2,5,-3]))  # UC_abyxz
    # Note that there is typo in Eq. (17) of arXiv:1912.02414
    MC = ncon((B, C),([-1,1,-3], [-2,1,-4]))
    DC = ncon((B, np.conjugate(U), A, np.conjugate(V), A),([1,-1,-5],[2,3,-2],[2,4,-4],[4,5,-3],[3,5,1]))

    Tmp = ncon((MC, UC),([-1,-2,1,2], [1,2,-3,-4,-5]))
    G, st, D = tensorsvd(Tmp,[0,1,2],[3,4],Dcut) 
    G = ncon((G, st),([-1,-2,-3,1], [1,-4]))

    Tmp2 = ncon((DC, G),([-1,-2,-3,1,2], [1,2,-4,-5]))
    A, st2, MCprime = tensorsvd(Tmp2,[0,1],[2,3,4],Dcut) 

    MCprime = ncon((st2, MCprime),([-1,1], [1,-2,-3,-4]))
    B, st3, C = tensorsvd(MCprime,[0,1],[2,3],Dcut)
    B = ncon((B, st3),([-1,-2, 1], [1,-3]))

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

    T = ncon((A, B, C, D),([-1,-3,1], [1,-5,2],[2,-6,3], [3,-4,-2]))
    # Same as np.einsum("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W) 
    norm = np.max(T)
    CU = np.log(norm)

    A  /= np.sqrt(np.sqrt(norm))
    B  /= np.sqrt(np.sqrt(norm))
    C  /= np.sqrt(np.sqrt(norm))
    D  /= np.sqrt(np.sqrt(norm))

 
    for iter in range (Niter):

        A, B, C, D = coarse_graining(A,B,C,D)  
        #T_ijklmn = A_ika * B_amb * C_bnc * D_clj 
        T = ncon((A, B, C, D),([-1,-3,1], [1,-5,2],[2,-6,3], [3,-4,-2])) 
        norm = np.max(T)

        A  /= np.sqrt(np.sqrt(norm))
        B  /= np.sqrt(np.sqrt(norm))
        C  /= np.sqrt(np.sqrt(norm))
        D  /= np.sqrt(np.sqrt(norm))

        CU += np.log(norm)/(8**(Niter))
        print ("Finished", iter+1, "steps of CG")
        
        if iter == Niter-1:
            Z = ncon((A, B, C, D, A, B, C, D),([4,6,1], [1,8,2],[2,9,3], [3,7,5], [4,6,10], [10,8,11],[11,9,12], [12,7,5]))   
            Free = -Temp*(CU + (np.log(Z)/(8*(Niter))))
            print ("Free energy = ", round(Free,4))

        
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 

