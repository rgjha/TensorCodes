# Tensor formulation of a classical statistical 2d model
# This implements the internet way. 
# Checking if it reproduces. 


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
from doTNR import doTNR     # See doTNR.py for details !

if len(sys.argv) < 3:
  print("Usage:", str(sys.argv[0]), "<Temperature, T>  <h>" )
  sys.exit(1)



Temp =  float(sys.argv[1])
beta = float(1.0/Temp)
h =  float(sys.argv[2])

D=25
D_cut=25
Niters=6
Ns = int(2**((Niters)))
Nt = Ns  
vol = Ns**2
numlevels = Niters # number of coarse-grainings
norm_all = [0 for x in range(numlevels+1)]

# 1.04 -0.5818237291118312 with D=25 and Niters=6


if D%2 == 0:
    print ("D must be odd for now")
    sys.exit(1) 


Dn = int(D/2.0)

startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 


A = np.zeros([D])     
L = np.zeros([D])              
ATNR = [0 for x in range(numlevels+1)];
ATNRnorm = [0 for x in range(numlevels+1)]; 
sXcg = [0 for x in range(numlevels+1)]; 


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



def coarse_graining(matrix, in2):

    T = matrix  
    TI = in2 
    d = D**2

    print ("Iteration", int(count+1), "of" , numlevels) 
    Za = ncon((T, T),([-1,1,2,-2], [-3,1,2,-4])) # * 
    Zb = ncon((T, T),([-1,1,-2,2], [-3,1,-4,2])) # * 
    MMdag_prime = ncon((Za, Zb),([-1,1,-3,2], [-2,1,-4,2])) # * 
    MMdag_prime = MMdag_prime.reshape(D**2, D**2) 

    # MMdag_prime = ncon((T,T,T,T),([-1,1,2,3], [-3,1,2,4], [-2,5,3,6], [-4,5,4,6]))
    # Above line is equivalent to three * marked lines. 
    # But at least 13 times slower! 

    w, U = LA.eigh(MMdag_prime)
    idx = w.argsort()[::-1]
    s1 = w[idx]
    U = U[:,idx] 

    if np.size(U,1) > D_cut: 

        s = s1[:D_cut] 
        U = U[:,:D_cut]  
 
    U = U.reshape(D,D,D)
    T =  ncon((U, T, T, U),([1,2,-1], [1,3,-3,4], [2,5,4,-4], [3,5,-2]))
    TI = ncon((U, TI, T, U),([1,2,-1], [1,3,-3,4], [2,5,4,-4], [3,5,-2]))
    val = LA.norm(T) 


    return T, TI, val 



def CG_net(matrix, in2):

    T = matrix  
    TI = in2 
    d = D**2

    A = ncon([T,T],[[-2,-3,-4,1],[-1,1,-5,-6]])
    #U, s, V = tensorsvd(A,[0,1],[2,3,4,5],D) 
    MMdag_prime = np.transpose(A,[0,1]+[2,3,4,5])
    w, U = LA.eigh(MMdag_prime)
    idx = w.argsort()[::-1]
    s1 = w[idx]
    U = U[:,idx] 

    if np.size(U,1) > D_cut: 

        s = s1[:D_cut] 
        U = U[:,:D_cut] 

    A = ncon([U,A,U],[[1,2,-1],[1,2,-2,3,4,-4],[4,3,-3]])
    B = ncon([TI,T],[[-2,-3,-4,1],[-1,1,-5,-6]])    
    B = ncon([U,B,U],[[1,2,-1],[1,2,-2,3,4,-4],[4,3,-3]])
    
    AA = ncon([A,A],[[-1,-2,1,-6],[1,-3,-4,-5]])
    #U, s, V = tensorsvd(AA,[1,2],[0,3,4,5],D)  

    MMdag_prime = np.transpose(AA,[1,2]+[0,3,4,5])
    w, U = LA.eigh(MMdag_prime)
    idx = w.argsort()[::-1]
    s1 = w[idx]
    U = U[:,idx] 

    if np.size(U,1) > D_cut: 

        s = s1[:D_cut] 
        U = U[:,:D_cut]

    AA = ncon([U,AA,U],[[1,2,-2],[-1,1,2,-3,4,3],[3,4,-4]])  
    BA = ncon([B,A],[[-1,-2,1,-6],[1,-3,-4,-5]])
    BA = ncon([U,BA,U],[[1,2,-2],[-1,1,2,-3,4,3],[3,4,-4]])  
    
    maxAA = np.max(AA)
    AA = AA/maxAA #divides over largest value in the tensor
    BA = BA/maxAA
        
    return AA, BA, maxAA




def get_tensor():

    for i in range (-Dn,Dn+1):
        L[i+Dn] = np.sqrt(sp.special.iv(i, beta))
 
    out = ncon((L, L, L, L),([-1],[-2],[-3],[-4])) # Alt: T = np.einsum("i,j,k,l->ijkl", L, L, L, L)
    for l in range (-Dn,Dn+1):
        for r in range (-Dn,Dn+1):
            for u in range (-Dn,Dn+1):
                for d in range (-Dn,Dn+1):
                    index = l+u-r-d
                    out[l+Dn][r+Dn][u+Dn][d+Dn] *= sp.special.iv(index, betah)

    return out  



def get_site_mag():

    for i in range (-Dn,Dn+1):
        L[i+Dn] = np.sqrt(sp.special.iv(i, beta))
 
    out = ncon((L, L, L, L),([-1],[-2],[-3],[-4])) # Alt: T = np.einsum("i,j,k,l->ijkl", L, L, L, L)
    for l in range (-Dn,Dn+1):
        for r in range (-Dn,Dn+1):
            for u in range (-Dn,Dn+1):
                for d in range (-Dn,Dn+1):
                    index = l+u-r-d
                    out[l+Dn][r+Dn][u+Dn][d+Dn] *= beta*0.50*(sp.special.iv(index-1, beta*h) + sp.special.iv(index+1, beta*h))

    return out 



if __name__ == "__main__":

 
    betah=beta*h
    T = get_tensor()
    TI = get_site_mag()
    print ("Norm", LA.norm(TI))

    norm = LA.norm(T)
    T /= norm 
    TI /= norm 


    i = 0
    Z = ncon([T,T,T,T],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
    f_i = -Temp*(np.log(Z))/(4)
    C = 0
    N = 1
    
    
    a, s, maxAA = CG_net(T, TI)
    C = np.log(maxAA)+4*C
    N *= 4.
    if i >= 10:
        Z = ncon([T,T,T,T],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
        f = -Temp*(np.log(Z)+4*C)/(4*N)
        delta_f = np.abs((f - f_i)/f)

        while delta_f > 0.000001:
            f_i = f
            i += 1
   
        Z = ncon([T,T,T,T],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
        f = -temp*(np.log(Z)+4*C)/(4*N)
            
        
    print ("free is", f)
    #print (Temp, -lnZ/(vol*beta))
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))





    
