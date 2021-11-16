import sys
import math
from math import sqrt
import numpy as np
import scipy as sp                  
from scipy import special
from numpy import linalg as LA
from numpy import ndarray
import time 
import datetime 
from ncon import ncon

if len(sys.argv) < 2:
  print("Usage:", str(sys.argv[0]), "<Temperature, T>")
  sys.exit(1)

# For T=2.0, f_2d_Ising = -1.7455677143228514
# From Ref: https://github.com/under-Peter/TensorNetworkAD.jl


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
print ("-----------------------------------------------------------------")

 
Temp =  float(sys.argv[1])
beta = float(1.0/Temp)
chi=21
D=21
#A = np.zeros([D, D, D, D])    
C = np.ones([chi, chi])  
C /= 2.0           
T = np.ones([chi, D, chi])
T /= 2.0 

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)

Dbond_XY = 21 
L = np.zeros([Dbond_XY]) 

Dn = int(Dbond_XY/2.0)
betah= beta*0.0 


##############################
def dagger(a):

    return np.transpose(a).conj() 
##############################



def get_tensor():

    for i in range (-Dn,Dn+1):
        L[i+Dn] = np.sqrt(sp.special.iv(i, beta))
 
    out = ncon((L, L, L, L),([-1],[-2],[-3],[-4])) 
    # Alt: T = np.einsum("i,j,k,l->ijkl", L, L, L, L)
    for l in range (-Dn,Dn+1):
        for r in range (-Dn,Dn+1):
            for u in range (-Dn,Dn+1):
                for d in range (-Dn,Dn+1):
                    index = l+u-r-d
                    out[l+Dn][r+Dn][u+Dn][d+Dn] *= sp.special.iv(index, betah)

    return out  


def Z2d_Ising():

    a = np.sqrt(np.cosh(beta))
    b = np.sqrt(np.sinh(beta)) 
    W = np.array([[a,b],[a,-b]]) 
    out = np.einsum("ia, ib, ic, id  -> abcd", W, W, W, W)
    return out
 

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

    #X = sparse_random(100, 100, density=0.01, format='csr', random_state=42)
    #svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    #svd.fit(X)
    #U, s, V = randomized_svd(T, n_components=D+10, n_iter=10,random_state=None)
    
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


##############################

if __name__ == "__main__":
 

    #A = Z2d_Ising()
    A = get_tensor() 
    Anew = A.transpose(1,2,3,0)
    if np.allclose(A,Anew) == False:
        Anew2 = dagger(A.transpose(2,3,0,1))
        print ("Left/right Hermitian symmetry ? ", np.allclose(A,Anew2))
        sys.exit(1)
    else:
        print ("Rotationally symmetric A tensor") 

    norm = LA.norm(A)
    A /= norm  

    # CTRG
    # https://github.com/carrasqu/corner_transfer_renormalization_group/blob/master/ctmrgloop.m
    # https://github.com/under-Peter/CornerTransferMethods.jl
    # https://github.com/under-Peter/CornerTransferMethods.jl/blob/master/src/isingmpo.jl
    print (np.shape(A))
    #B=ncon([A,sigmaZ,sigmaZ],[[-1 -2 2 1],[-4 1],[-3 2]])
    #B = ncon([A,sigmaZ],[[-1,-2,-3,1],[-4,1]])  
    #B = np.einsum('ijkp,lp->ijkl', A, sigmaZ)


    for i in range (5):

      Cp = ncon([C,T,T,A],[[1,2],[-1,3,1],[2,4,-4],[-2,-3,4,3]])  
      Tp = ncon([T,A],[[-1,1,-5],[-3,-4,1,-2]]) 
      U, s, V = tensorsvd(Cp,[0,1],[2,3],10) 
      Cp = ncon([Cp,U,V],[[1,2,3,4],[1,2,-1],[-2,3,4]]) 
      Tp = ncon([Tp,U,V],[[1,2,-2,3,4],[1,2,-1],[-3,3,4]]) 
      Cp = (Cp + Cp.T)
      Tp = (Tp + Tp.T)
      Cp /= np.max(Cp)
      Tp /= np.max(Tp)

    print (np.shape(Cp), np.shape(Tp), np.shape(A))
    psi=ncon([Cp,Cp,Cp,Cp,Tp,Tp,Tp,Tp,Tp,Tp,Tp,Tp,A,A,A],[[1,2],[6,7],[17,18],[20,14],[2,3,4],[4,5,6],\
    [7,8,9],[13,12,1],[14,15,13],[9,-2,17],[18,-1,19],[19,16,20],[-3,8,5,10],[11,10,3,12],[16,-4,11,15]])  

    #print (A)
    #print (B)
    #mag=ncon([psi,B],[[1,2,3,4],[1,2,3,4]])
    partition=ncon([psi,A],[[1,2,3,4],[1,2,3,4]])
    #magnetization = mag/partition
    print (partition)
    print ("log (part) is", -np.log(partition))
    #print ("M is", magnetization)




            
#print ("Free energy density =", (-lnZ/vol)) 
print ("-----------------------------------------------------------------")           
print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

