# Tensor formulation of a classical statistical 2d model
# Calculate free energy, internal energy, 
# susceptibility of free energy, magnetization, critical exponent. 
# In progress!  March 4, 2019
# Restarted: Nov 3, 2019

# VAR = Sqrt[BesselI[l, a]]*Sqrt[BesselI[r, a]]*Sqrt[BesselI[u, a]]*Sqrt[BesselI[d, a]]*BesselI[(l + u - r - d), a*z]; 
# D[VAR, z] 1/VAR


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

D=35
D_cut=35
Niters=3
Ns = int(2**((Niters)))
Nt = Ns  
vol = Ns**2
numlevels = Niters # number of coarse-grainings
norm_all = [0 for x in range(numlevels+1)]


if D%2 == 0:
    print ("D must be odd for now")
    sys.exit(1) 


Dn = int(D/2.0)

##### Set bond dimensions and options
chiM = 25
chiS = 25
chiU = 25
chiH = 25      
chiV = 25
#Increasing these makes it more accurate!        


###### Initialize tensor lists
SPerrs = np.zeros((numlevels,4))
qC = [0 for x in range(numlevels)]
sC = [0 for x in range(numlevels)]
uC = [0 for x in range(numlevels)]
yC = [0 for x in range(numlevels)]
vC = [0 for x in range(numlevels)]
wC = [0 for x in range(numlevels)]



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

    w, U = LA.eigh(MMdag_prime)
    idx = w.argsort()[::-1]
    s1 = w[idx]
    U = U[:,idx] 

    if np.size(U,1) > D_cut: 

        s = s1[:D_cut] 
        U = U[:,:D_cut]  
 
    U = U.reshape(D,D,D)
    T =  ncon((U, T, T, U),([1,2,-1], [1,3,-3,4], [2,5,4,-4], [3,5,-2]))
    TI = ncon((U, T, TI, U),([1,2,-1], [1,3,-3,4], [2,5,4,-4], [3,5,-2]))
    val = LA.norm(T) 


    return T, TI, val 




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
    nc_pure = (2**((2*numlevels))) * np.log(norm)
    count = 0.0 
    eps = 0.0  


    if flag == 0:

        for i in range (0,Niters):

            T, TI, norm = coarse_graining(T, TI) 
            count += 1  
            T /= norm 
            TI /= norm 
            
            nc_pure += (2**((2*numlevels)-count)) * np.log(norm)
 

        T = np.einsum("iikl->kl", T)
        TI  = np.einsum("iikl->kl", TI)


        for i in range (0, Niters):

            T = np.dot(T,T)
            TI  = np.dot(TI,TI)
            norm = np.trace(T)
            T /= norm 
            #TI  /= norm 
            #print ('TI norms', LA.norm(TI)) 



        trT = np.einsum("ii", T)
        trTI = np.einsum("ii", TI)
        print ("TI", trTI)
        print ("T", trT)
        print ("Alt. mag", trTI/trT)
        lnZ = np.log(trT) + nc_pure  

        #trTimpure = np.einsum("ii", TI)
        #print ("Trace imp.",trTimpure)


        print (Temp, -lnZ/(vol*beta))




    if flag == 1: 

        numlevels = int(numlevels/2.0)

        TI /= LA.norm(TI)

        ATNR[0] = T 
        ATNRnorm[0] = norm 
        sXcg[0] = TI 

        for k in range(numlevels):
            print ("Iteration", int(k+1), "of" , numlevels)
            ATNR[k+1], qC[k], sC[k], uC[k], yC[k], vC[k], wC[k], ATNRnorm[k+1], SPerrs[k,:] = \
            doTNR(ATNR[k],[chiM,chiS,chiU,chiH,chiV], 1e-8, 1000, 100, True, 0.1)
            Tmp = ATNR[k+1] 
            norm = ATNRnorm[k+1]


            ''' 
            1st argument: Eigenvalue threshold for automatic truncate indication of indices
            (increase means less accurate!)
            2nd argument: Maximum number of iterations in disentangler optimization
            3rd argument: Minimum number of iterations in disentangler optimization
            4th argument: Display information during optimization
            5th argument: Threshold for relative error change to stop disentangler optimization 
            '''


        #Volume = 2**(2*np.int64(np.array(range(1,18)))-2)   # 4^n, where n runs from 0 ... max 
        FreeEnergy = np.zeros(numlevels);
        
        for k in range(1,numlevels+1): 
            Hgauge = ncon([vC[k-1],vC[k-1]],[[1,2,-1],[2,1,-2]])
            Vgauge = ncon([wC[k-1],wC[k-1]],[[1,2,-1],[2,1,-2]])
            FreeEnergy[k-1] = -1.0*(sum((4**np.int64(np.array(range(k,-1,-1))))*np.log(ATNRnorm[:(k+1)]))+ \
            np.log(ncon([ATNR[k],Hgauge,Vgauge],[[1,3,2,4],[1,2],[3,4]])))


        for k in range(1,numlevels+1): 
            Hgauge = ncon([vC[k-1],vC[k-1]],[[1,2,-1],[2,1,-2]])
            Vgauge = ncon([wC[k-1],wC[k-1]],[[1,2,-1],[2,1,-2]])
            FreeEnergy[k-1] = -1.0*(sum((4**np.int64(np.array(range(k,-1,-1))))*np.log(ATNRnorm[:(k+1)]))+ \
            np.log(ncon([ATNR[k],Hgauge,Vgauge],[[1,3,2,4],[1,2],[3,4]])))

        ##### Change gauge on disentanglers 'u'
        a = int(np.sqrt(chiM))
        gaugeX = np.eye(chiM).reshape(a,a,a,a).transpose(1,0,2,3).reshape(chiM,chiM)
        upC = [0 for x in range(numlevels)]
        upC[0] = ncon([uC[0],gaugeX],[[-1,1,-3,-4],[1,-2]])
        for k in range(1,numlevels):
            uF,sF,vhF = LA.svd(ncon([wC[k-1],wC[k-1]],[[1,2,-1],[2,1,-2]]))
            gaugeX = uF @ vhF
            upC[k] = ncon([uC[k],gaugeX],[[-1,1,-3,-4],[1,-2]])



        for k in range(numlevels):
            sXcg[k+1] = ncon([sXcg[k],upC[k],upC[k],wC[k],wC[k],wC[k],wC[k]],\
            [[3,4,1,2],[1,2,6,9],[3,4,7,10],[5,6,-1],[9,8,-2],[5,7,-3],[10,8,-4]])
    
        dtemp = LA.eigvalsh(sXcg[numlevels].reshape((sXcg[numlevels].shape[0])**2,(sXcg[numlevels].shape[2])**2))
        ExpectX = max(abs(dtemp))
        print ("Magnetization, ", ExpectX)



        print ("Temperature is", Temp, "and free energy using TNR is", FreeEnergy[numlevels-1]/(beta*(4**numlevels)))

    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
