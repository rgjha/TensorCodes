# Tensor formulation of a classical statistical 2d model
# Calculate free energy, internal energy, 
# susceptibility of free energy, magnetization, critical exponent. 
# In progress!  March 4, 2019

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



if len(sys.argv) < 4:
  print("Usage:", str(sys.argv[0]), "<Temperature, T>  <h>  <TNR is 1, HOTRG is 0> ")
  sys.exit(1)


Temp =  float(sys.argv[1])
h =  float(sys.argv[2])
flag = int(sys.argv[3])
beta = float(1.0/Temp)

# Usual method: # D=22 is 19sec, D=27 is 262 seconds 
# Improvement by never explicitly constructing M --> D=27 is 8 seconds, 
# D=32 is 28 sec, D=36 is 58 sec, D=40 takes about 160 sec. 

D=30
D_cut=30
Niters=8
Ns = int(2**((Niters)))
Nt = Ns  
vol = Ns**2
numlevels = Niters # number of coarse-grainings


##### Set bond dimensions and options
chiM = 12
chiS = 12
chiU = 12
chiH = 12       
chiV = 12        


###### Initialize tensor lists
SPerrs = np.zeros((numlevels,4))
qC = [0 for x in range(numlevels)]
sC = [0 for x in range(numlevels)]
uC = [0 for x in range(numlevels)]
yC = [0 for x in range(numlevels)]
vC = [0 for x in range(numlevels)]
wC = [0 for x in range(numlevels)]


OPTS_dtol = 1e-5 # eigenvalue threshold for automatic truncate indication of indices
OPTS_disiter = 2000 # maximum number of iterations in disentangler optimization
OPTS_miniter = 200 # minimum number of iterations in disentangler optimization
OPTS_dispon = True # display information during optimization
OPTS_convtol = 0.01 # threshold for relative error change to stop disentangler optimization


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 


A = np.zeros([D])     
L = np.zeros([D])              
B = np.zeros([2*D])
ATNR = [0 for x in range(numlevels+1)];
ATNRnorm = [0 for x in range(numlevels+1)]; 


##############################
def dagger(a):

    return np.transpose(a).conj()
##############################



def coarse_graining(matrix, eps, nc, count):

    T = matrix  
    d = D**2

    print ("Iteration", int(count+1), "of" , numlevels) 
    Za = ncon((T, T),([-1,1,2,-2], [-3,1,2,-4]))
    Zb = ncon((T, T),([-1,1,-2,2], [-3,1,-4,2]))
    MMdag_prime = ncon((Za, Zb),([-1,1,-3,2], [-2,1,-4,2]))
    MMdag_prime = MMdag_prime.reshape(D**2, D**2)    
    w, U = LA.eigh(MMdag_prime)
    idx = w.argsort()[::-1]
    s1 = w[idx]
    U = U[:,idx] 

    if np.size(U,1) > D_cut: 

        s = s1[:D_cut] 
        U = U[:,:D_cut]  

    else:
        s = s1
        eps = 1.0 - (sum(s)/sum(s1))
        print ("Error is", eps)

    count += 1 
    U = U.reshape(D,D,D)
    M_new =  ncon((U, T, T, U),([1,2,-1], [1,3,-3,4], [2,5,4,-4], [3,5,-2]))
    norm = LA.norm(M_new)
    nc += (2**((2*numlevels)-count)) * np.log(norm)

    if norm != 0:
        T = M_new/norm 

    else: 
      T = M_new
    
    return T, eps, nc, count  


if __name__ == "__main__":

    for i in range (0,D):

        L[i] = np.sqrt(sp.special.iv(i, beta))
 
    T = ncon((L, L, L, L),([-1],[-2],[-3],[-4])) # Alt: T = np.einsum("i,j,k,l->ijkl", L, L, L, L)
    betah=beta*h

    for l in range (0,D):
        for r in range (0,D):
            for u in range (0,D):
                for d in range (0,D):

                    index = l+u-r-d
                    
                    if index != 0:
                        T[l][r][u][d] = 0.0  

    norm = LA.norm(T)
    T /= norm 
    nc = (2**((2*numlevels))) * np.log(norm)
    count = 0.0 
    eps = 0.0  


    if flag == 0:

        for i in range (0,Niters):

            T, eps, nc, count = coarse_graining(T, eps, nc, count)   
   
        T = np.einsum("iikl->kl", T)

        for i in range (0, Niters):

            T = np.dot(T, T)
            T /= np.trace(T)


        trT = np.einsum("ii", T)
        #trT = trT * np.sqrt(vol)
        lnZ = np.log(trT) + nc  

        print ("Temperature is", Temp, " and free energy using HOTRG is", -lnZ/(vol*beta))



    if flag == 1: 

        numlevels = int(numlevels/2.0)

        ATNR[0] = T 
        ATNRnorm[0] = norm 

        for k in range(numlevels):
            print ("Iteration", int(k+1), "of" , numlevels)
            ATNR[k+1], qC[k], sC[k], uC[k], yC[k], vC[k], wC[k], ATNRnorm[k+1], SPerrs[k,:] = \
            doTNR(ATNR[0],[chiM,chiS,chiU,chiH,chiV], 1e-13, 3000, 100, True, 0.01)


        Volume = 2**(2*np.int64(np.array(range(1,18)))-2)   # 4^n, where n runs from 0 ... max 
        FreeEnergy = np.zeros(numlevels);
        
        for k in range(1,numlevels+1): 
            Hgauge = ncon([vC[k-1],vC[k-1]],[[1,2,-1],[2,1,-2]])
            Vgauge = ncon([wC[k-1],wC[k-1]],[[1,2,-1],[2,1,-2]])
            FreeEnergy[k-1] = -1.0*(sum((4**np.int64(np.array(range(k,-1,-1))))*np.log(ATNRnorm[:(k+1)]))+ \
            np.log(ncon([ATNR[k],Hgauge,Vgauge],[[1,3,2,4],[1,2],[3,4]])))/Volume[k] 

        print ("Temperature is", Temp, "and free energy using TNR is", FreeEnergy[numlevels-1])

    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
