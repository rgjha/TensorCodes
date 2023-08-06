# Tensor formulation of a classical statistical 2d model
# This proceeds by blocking simulatenously along both directions. 
# Reproduces free energy and magentization from arxiv:1309.04963
# D = 53 with Niter = 40 takes ~ 10 hours (The timing is on the Symmetry machine at PI)
# D = 55 gives memory error on Symmetry!
# Part of this code was used for https://arxiv.org/abs/2004.06314
# Fixed typos: 6 August 2023 following request over email from ZXM

import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
from scipy import special
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
from scipy.sparse import random as sparse_random
import time
import datetime 
#from ncon import ncon # No longer using this
from opt_einsum import contract # https://pypi.org/project/opt-einsum/

if len(sys.argv) < 4:
  print("Usage:", str(sys.argv[0]), "<Temperature, T>  <h> <Dbond> " )
  sys.exit(1)

Temp =  float(sys.argv[1])
beta = float(1.0/Temp)
h =  float(sys.argv[2])
D = int(sys.argv[3])
D_cut = D 

Niters = 40
Ns = int(2**((Niters)))
Nt = Ns  
vol = Ns**2   
numlevels = Niters
norm_all = [0 for x in range(numlevels+1)]

if D%2 == 0:
    print ("D must be odd for now")
    sys.exit(1) 


Dn = int(D/2.0)

startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 


A = np.zeros([D])     
L = np.zeros([D])               


def tensorsvd(input,left,right,D):
    
    '''Reshape input tensor into a matrix with first index corresponding
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


##############################
def zeta_proxy(A):
    w, U = LA.eigh(A) 
    idx = w.argsort()[::-1]
    s = w[idx]
    return 1.0/(np.log(s[0]/s[1]))


##############################
def dagger(a):

    return np.transpose(a).conj()
##############################

def CG_step(matrix, in2):

    t = matrix  
    tim = in2 

    AAdag = contract('jabe,iecd,labf,kfcd->ijkl', t, t, t, t)
    U, s, V = tensorsvd(AAdag,[0,1],[2,3],D_cut)  

    #A = ncon([U,T,T,U],[[1,2,-1],[2,-2,4,3],[1,3,5,-4],[5,4,-3]])
    A = contract('abi,bjdc,acel,edk->ijkl', U, t, t, U)
    B = contract('abi,bjdc,acel,edk->ijkl', U, tim, t, U) 

    #B = ncon([U,TI,T,U],[[1,2,-1],[2,-2,4,3],[1,3,5,-4],[5,4,-3]])

    #AAdag = ncon([A,A,A,A],[[1,-1,2,3],[2,-2,4,5],[1,-3,6,3],[6,-4,4,5]]) # Reuse old memory allocated
    AAAAdag = contract('aibc,bjde,akfc,flde->ijkl',A,A,A,A)

    U, s, V = tensorsvd(AAAAdag,[0,1],[2,3],D_cut) 

    AA = contract('abj,iacd,cbke,del->ijkl', U, A, A, U) 
    #AA = ncon([U,A,A,U],[[1,2,-2],[-1,1,3,4],[3,2,-3,5],[4,5,-4]])
    #BA2 = ncon([U,B,A,U],[[1,2,-2],[-1,1,4,3],[4,2,-3,5],[3,5,-4]]) 
    BA = contract('abj,iadc,dbke,cel->ijkl', U, B, A, U) 
    #BA = ncon([U,B,A,U],[[1,2,-2],[-1,1,3,4],[3,2,-3,5],[4,5,-4]]) 
    # BA and BA2 are same!
    
    maxAA = np.max(AA)
    AA = AA/maxAA # Normalize by largest element of the tensor
    BA = BA/maxAA
        
    return AA, BA, maxAA


def get_tensor():

    for i in range (-Dn,Dn+1):
        L[i+Dn] = np.sqrt(sp.special.iv(i, beta))
 
    #out = ncon((L, L, L, L),([-1],[-2],[-3],[-4])) 
    out = contract("i,j,k,l->ijkl", L, L, L, L)
    for l in range (-Dn,Dn+1):
        for r in range (-Dn,Dn+1):
            for u in range (-Dn,Dn+1):
                for d in range (-Dn,Dn+1):
                    index = l+u-r-d
                    out[l+Dn][r+Dn][u+Dn][d+Dn] *= sp.special.iv(index, beta*h)

    if D_cut > 47:
        return out.astype(np.float32)
    else: 
        return out 



def get_site_mag():

    for i in range (-Dn,Dn+1):
        L[i+Dn] = np.sqrt(sp.special.iv(i, beta))
 
    #out = ncon((L, L, L, L),([-1],[-2],[-3],[-4])) 
    out = contract("i,j,k,l->ijkl", L, L, L, L)
    for l in range (-Dn,Dn+1):
        for r in range (-Dn,Dn+1):
            for u in range (-Dn,Dn+1):
                for d in range (-Dn,Dn+1):
                    index = l+u-r-d
                    out[l+Dn][r+Dn][u+Dn][d+Dn] *= 0.50 * (sp.special.iv(index-1, beta*h) + sp.special.iv(index+1, beta*h))

    if D_cut > 47:
        return out.astype(np.float32)
    else: 
        return out


if __name__ == "__main__":

    T = get_tensor()
    Tim = get_site_mag()

    norm = np.max(T)
    T /= norm 
    Tim /= norm 

    #Z = ncon([T,T,T,T],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
    C = 0.0
    N = 1.0
    C = np.log(norm)

    for i in range (Niters):

        
        T, Tim, norm = CG_step(T, Tim)
        N *= 4.0
        C += (math.log(norm)/4**(i+1))
         
        if i == Niters-1:

            #Z1 = ncon([T,T],[[1,-1,2,-2],[2,-3,1,-4]])
            Z1 = contract('aibj,bkal->ijkl', T, T)
            #Z = ncon([Z1,Z1],[[1,2,3,4],[2,1,4,3]])
            Z = contract('abcd,badc->''', Z1, Z1)
            Free = -Temp*(C + (np.log(Z)/(4**(Niters))))
            #P = ncon([Tim,T],[[1,-1,2,-2],[2,-3,1,-4]])
            P = contract('aibj,bkal->ijkl', Tim, T)
            #P = ncon([P,Z1],[[1,2,3,4],[2,1,4,3]]) 
            P = contract('abcd,badc->''', P, Z1) 
            mag = (P/Z)
            print ("Free energy density = ", Free)
            print ("Magnetization = ", mag)

               
    f=open("mag_data.txt", "a+")    
    f.write("%4.10f \t %4.10f \t %4.10f \t %2.0f \t %2.0f \t %4.13f \n" % (Temp, Free, mag, Niters, D_cut, h)) 
    f.close()         
    print (Temp,h,Free,mag)
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
