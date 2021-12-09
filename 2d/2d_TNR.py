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
from scipy.stats import unitary_group
from scipy.stats import ortho_group
from packages import doTNR # Fix me if needed


if len(sys.argv) < 2:
  print("Usage:", str(sys.argv[0]), "<Verbose or not>")
  sys.exit(1)


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
print ("-----------------------------------------------------------------")

representation = [0, 1]                   # By definition : 2r 
dim = [x+1 for x in representation]       # 2r + 1 
rep_max = int(max(representation))                 
N_r = int(sum(np.square(dim))) 
N_m = int(max(dim))
  
kappa = 2.0                            # Coupling 
Niters = 14                            
beta = 0.0
numlevels = Niters # number of coarse-grainings


##### Set bond dimensions and options
chiM = 12
chiS = 12
chiU = 12
chiH = 12       
chiV = 5        


###### Initialize tensor lists
SPerrs = np.zeros((numlevels,4))
qC = [0 for x in range(numlevels)]
sC = [0 for x in range(numlevels)]
uC = [0 for x in range(numlevels)]
yC = [0 for x in range(numlevels)]
vC = [0 for x in range(numlevels)]
wC = [0 for x in range(numlevels)]



OPTS_dtol = 1e-10 # eigenvalue threshold for automatic truncate indication of indices
OPTS_disiter = 2000 # maximum number of iterations in disentangler optimization
OPTS_miniter = 200 # minimum number of iterations in disentangler optimization
OPTS_dispon = True # display information during optimization
OPTS_convtol = 0.01 # threshold for relative error change to stop disentangler optimization

verbose = int(sys.argv[1]) 

                               
A = np.zeros([N_r, N_r])                
B = np.zeros([N_r, N_r, N_r, N_r])
ATNR = [0 for x in range(numlevels+1)];
ATNRnorm = [0 for x in range(numlevels+1)]; 


##############################
def index(a, b, c):

    return int((a * (a + 1) * ((2.0*a) + 1)/(6.0)) + (a+2)*(a/2.0) + (a+1)*b + c)
    # sum_{j=0}^{j=N} (j+1)^2 = (N+1)*(N+2)*(2*N+3)/(6.0) is used. 
##############################


##############################
def dagger(a):

    return np.transpose(a).conj()
##############################


##############################
def factorial(N):
    if N < 0:
        print ("N is ", N)
        raise ValueError("N is negative !!! ")
        return 0 
    if N+1 == N:
        raise OverflowError("N is too large !!!")
    result = 1
    factor = 2
    while factor <= N:
        result *= factor
        factor += 1
    return result
#####################################


##############################
def contract_reshape(A, B, dout):
    if dout < 0:
        raise ValueError("Dimension of matrix is negative !!! ")
        return 0

    dum = np.tensordot(A,B,axes=([3,2]))  # Note that 0=left, 1=right, 2=top, 3=bottom 
    dum = dum.transpose(0,3,1,4,2,5)
    out = dum.reshape(dout, dout, N_r, N_r)  

    return out
#####################################


#####################################
# Returns Clebsch-Gordon coefficients
# Alternative : from sympy.physics.quantum.cg import CG 

def CGC(j1, m1, j2, m2, j, m):

    if (m == m1+m2) and (abs(j1 - j2) <= j <= (j1 + j2)) and (-j <= m <= j) and (-j1 <= m1 <= j1) and (-j2 <= m2 <= j2):

        A = sqrt(float((2*j + 1)*factorial(j + j1 - j2)*factorial(j - j1 + j2)*factorial(j1 + j2 - j))/(factorial(j1 + j2 + j + 1)))
        B = sqrt(factorial(j + m)*factorial(j - m)*factorial(j1 - m1)*factorial(j1 + m1)*factorial(j2 - m2)*factorial(j2 + m2))
        C = A*B

        dum = 0

        lim = int(math.floor(abs(j + m + 1)))
        for k in range(0, lim+2):    

            if (j1 + j2 >= j+k) and (j1 >= m1+k) and (j2 + m2 >= k) and (j + m1 + k >= j2) and (j + k >= j1 + m2):
                dum += ((-1)**(k))/(factorial(k) \
                *factorial(j1 + j2 - j - k)*factorial(j1 - m1 - k)*factorial(j2 + m2 - k)*factorial(j - j2 + m1 + k) \
                *factorial(j - j1 - m2 +k))
            else:
                dum = dum   

        C *= dum
        return C

    else:
        return 0
##############################


##############################
def Fr(a, b):
    if b < 0 or a < 0:
        raise ValueError(" a or b is negative !!! ")
        return 0
    elif b==0 and a==1:
        return 0
    elif b==0 and a==0:
        return 2.0 * (a+1.0) * 0.50   # lim besselj[1,x]/x as x->0 = 0.5
    else:
        return 2.0 * (a+1.0) * sp.special.iv((a+1.0), b)/b 
##############################


##############################
def make_tensorA(rep):

    for r_l in rep:
        for r_r in rep:

            m_a = []
            m_b = []

            if r_l == 0:
                m_a.append(0) 
            else:
                for x in [-r_l, r_l]:
                    m_a.append(x/2.0)    

            if r_r == 0:
                m_b.append(0) 
            else:
                for x in [-r_r, r_r]:
                    m_b.append(x/2.0) 

            for m_al in m_a:
                for m_ar in m_a:
                    for m_bl in m_b:
                        for m_br in m_b:

                            k = index(r_l, m_al, m_ar)
                            l = index(r_r, m_bl, m_br)
  
                            for sigma in range(abs(r_l-r_r), abs(r_l+r_r)+1, 2): 
                                CG1 = CGC((r_l/2.0), m_al, (sigma/2.0), (m_bl - m_al), (r_r/2.0), m_bl)
                                CG2 = CGC((r_l/2.0), m_ar, (sigma/2.0), (m_bl - m_al), (r_r/2.0), m_br) 
                                A[k][l] += Fr((sigma), kappa) *  CG1  *  CG2 / (r_r + 1)
    return A  
##############################


##############################
def make_tensorB(rep):

    m_a = []
    m_b = []

    for r in rep:
        r_l = r_r = r_l = r_r = r 

        if r == 0:
                m_al = m_ar = m_bl = m_br = m_la = m_ra = m_rb = m_lb = 0
                j = index(r, m_al, m_lb)
                k = index(r, m_ra, m_br)
                l = index(r, m_al, m_ra)
                m = index(r, m_lb, m_br)
                B[j][k][l][m] = Fr(r, beta)    # 1st element
                
        else:
            
            for x in [-r, r]:
                m_a.append(x/2.0)
                m_b.append(x/2.0)

            for m_al in m_a:
                for m_bl in m_b:
                    for m_ar in m_a:
                        for m_br in m_b:

                            # Note that delta-function condition is imposed and 
                            # loops over other 4m's removed. 

                            j = index(r, m_al, m_bl)
                            k = index(r, m_ar, m_br)
                            l = index(r, m_al, m_ar)
                            m = index(r, m_bl, m_br)
                            B[j][k][l][m] = Fr(r, beta)

    return  B

##############################


if __name__ == "__main__":
 
    A = make_tensorA(representation)  # Link tensor  
    B = make_tensorB(representation)  # Plaquette tensor

    L = LA.cholesky(A)

    Linverse = LA.inv(L)

    T = np.einsum("pjkl, pa, jb, kc, ld", B, L, L, L, L) 
    norm = LA.norm(T)
    T /= norm 
    ATNR[0] = T 
    ATNRnorm[0] = norm  

    for k in range(numlevels):
        ATNR[k+1], qC[k], sC[k], uC[k], yC[k], vC[k], wC[k], ATNRnorm[k+1], SPerrs[k,:] = doTNR(T,[chiM,chiS,chiU,chiH,chiV], 1e-10, 2000, 100, True, 0.01)

    Volume = 2**(2*np.int64(np.array(range(1,18)))-2)   # 4^n, where n runs from 0 ... max 
    FreeEnergy = np.zeros(numlevels);
    for k in range(1,numlevels+1): 
        Hgauge = ncon([vC[k-1],vC[k-1]],[[1,2,-1],[2,1,-2]])
        Vgauge = ncon([wC[k-1],wC[k-1]],[[1,2,-1],[2,1,-2]])
        FreeEnergy[k-1] = -1.0*(sum((4**np.int64(np.array(range(k,-1,-1))))*np.log(ATNRnorm[:(k+1)]))+ \
        np.log(ncon([ATNR[k],Hgauge,Vgauge],[[1,3,2,4],[1,2],[3,4]])))/Volume[k] 
    

print ("Free energy using TNR is ", FreeEnergy[numlevels-1])   
print ("-----------------------------------------------------------------") 
#print ("Free energy for kappa=2, beta=0 from HOTRG was about -0.9282689470923198")         
print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))




