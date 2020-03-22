#!/usr/bin/python3
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
from numpy.linalg import matrix_power


if len(sys.argv) < 4:
  print("Usage:", str(sys.argv[0]), " beta " " kappa " " D_cut " )
  sys.exit(1)


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 

rmax = 0.50   

representation = [x for x in range (0, int(2.0*rmax)+1, 1)]  
dim = [x+1 for x in representation]   
rep_max = int(max(representation))                 
N_r = int(sum(np.square(dim))) 
N_m = int(max(dim))                              
Niters = 4  
Niters_time = Niters
c=0.17    # For c > 0.17 need to fix with Ns=6=Nt !
Ns = int(2**((Niters)))
Nt = int(2**((Niters_time))) 
vol = Ns*Nt
beta = float(sys.argv[1]) 
kappa = float(sys.argv[2])
D_cut = int(sys.argv[3])
nmax = 4    # Maximum order of Renyi entropy, n=1 is von Neuman 

if D_cut > 63:
    print ("It will take at least 11 minutes")

if D_cut > 69:
    print ("It will take at least 15 minutes")

if D_cut > 74:
    print ("It will take at least 21 minutes")

                               
A = np.zeros([N_r, N_r])                
B = np.zeros([N_r, N_r, N_r, N_r])
rho = [None] * nmax
S = [None] * nmax


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
    elif b==0 and a==2:
        return 0   # lim besselj[2,x]/x as x->0 = 0
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
                for x in range(-r_l, r_l+1, 2):
                    m_a.append(x/2.0)     

            if r_r == 0:
                m_b.append(0) 
            else:
                for x in range(-r_r, r_r+1, 2):
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
                B[j][k][l][m] = Fr(r, beta)   
                
        else:
            
            for x in [-r, r]:
                m_a.append(x/2.0)
                m_b.append(x/2.0)

            for m_al in m_a:
                for m_bl in m_b:
                    for m_ar in m_a:
                        for m_br in m_b:


                            j = index(r, m_al, m_bl)
                            k = index(r, m_ar, m_br)
                            l = index(r, m_al, m_ar)
                            m = index(r, m_bl, m_br)
                            B[j][k][l][m] = Fr(r, beta)

    return  B

##############################


def coarse_graining(matrix, eps, nc, count):

    T = matrix  
    d = int(min(D_cut**2, N_r**(2*(count+1))))

    print ("Iteration", int(count+1), "of" , Niters) 
    Za = ncon((T, T),([-1,1,2,-2], [-3,1,2,-4]))       
    Zb = ncon((T, T),([-1,1,-2,2], [-3,1,-4,2]))      
    MMdag_prime = ncon((Za, Zb),([-1,1,-3,2], [-2,1,-4,2])) 
    MMdag_prime = MMdag_prime.reshape(d, d)  


    w, U = LA.eigh(MMdag_prime)   # Slow alternative : U, s1, V = LA.svd(MMdag_prime)
    idx = w.argsort()[::-1]
    s1 = w[idx]
    U = U[:,idx] 


    if np.size(U,1) > D_cut: 

        s = s1[:D_cut] 
        U = U[:,:D_cut]  


    U = U.reshape(int(sqrt(np.size(U,0))),int(sqrt(np.size(U,0))),np.size(U,1))
    M_new =  ncon((U, T, T, U),([1,2,-1], [1,3,-3,4], [2,5,4,-4], [3,5,-2]))

    norm = LA.norm(M_new)

    if norm != 0:
        T = M_new/norm

    else: 
      T = M_new

    count += 1 

    return T, eps, nc, count  


def renyi(Rho, nmax): 

    for i in range (2,nmax+1):

        rho[i-2] = matrix_power(Rho, i)             # Raise to power "n" [\rho_A]^n 
        S[i-2] = math.log(np.trace(rho[i-2]))/(1-i) # Take log and divide by 1-n  
        # Check once TODO
    
    return S 

def von(Rho):
    u,v = LA.eig(Rho)
    chi = u.shape[0] 
    EE = 0
    for n in range (0 , chi):
        if u[n] > 0:
            EE += -u[n] * math.log(u[n].real)
    return EE 


if __name__ == "__main__":
 
    A = make_tensorA(representation)  # Link tensor  
    B = make_tensorB(representation)  # Plaquette tensor 

    L = LA.cholesky(A)

    Linverse = LA.inv(L)

    T = np.einsum("pjkl, pa, jb, kc, ld", B, L, L, L, L) 
    norm = LA.norm(T)
    T /= norm 

    Tprime = np.einsum("pjkl, kc, ld -> pjcd", B, L, L)
    normprime = LA.norm(Tprime)
    Tprime /= normprime 
     

    count = 0.0 
    eps = 0.0  

    nc = (2**((2*Niters))) * np.log(norm)

    for i in range (0,Niters-1):

        T, eps, nc, count = coarse_graining(T, eps, nc, count)   

    T = T.transpose(2,3,1,0)    # Rotate CCW 90 degrees ** 
    Tnew = ncon((T, T),([1,2,-1,-2], [2,1,-3,-4])) 

    Tnew = Tnew.transpose(0,2,1,3)  # Combine correct indices (top and bottom) 
    Tnew = Tnew.reshape(D_cut**2, D_cut**2)


    for i in range (0, Niters_time):

        Tnew = np.matmul(Tnew,Tnew)              # 0. Raise over the time slices
        Tnew /= np.trace(Tnew)


    Tnew = Tnew/np.trace(Tnew)                   # 1. Normalize to recognize (T)^Nt as \rho
    Tnew = Tnew.reshape(D_cut,D_cut,D_cut,D_cut) # 2. Expose indices for A & B respectively 
    rho_A = np.einsum("klii", Tnew)              # 3. Trace over environment/ subsystem B, or A as chosen
    # Trace of \rho should be ~1 already, no need to divide!  

    # Check TODO

    EE = renyi(rho_A, nmax)
    VN =  von(rho_A)
    print (beta, VN.real, EE[0]) 


    print ("Finished",count+1,"C.G. steps with",D_cut,"states, " "kappa =", kappa, "with rmax =", rmax , "and beta", beta)          
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print ("-----------------------------------------------------------------") 










