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
  print("Usage:", str(sys.argv[0]), "<Verbose or not> " "nmax " "kappa" )
  sys.exit(1)


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
print ("-----------------------------------------------------------------")

representation = [0, 1]                    
dim = [x+1 for x in representation]   
rep_max = int(max(representation))                 
N_r = int(sum(np.square(dim))) 
N_m = int(max(dim))                              
Niters = 3
Ns = int(2**((Niters)))
Nt = Ns                                  # Number of time-slices
vol = Ns**2
D_cut = 39
beta = 0.01*vol 

#n = 2                 

if D_cut <= N_r**2:
  print("Usage: D_cut must be greater than N_r**2 for now")
  sys.exit(1)


verbose = int(sys.argv[1]) 
nmax = int(sys.argv[2])    # Maximum order of Renyi entropy, n=1 is von Neumann
kappa = float(sys.argv[3])
                               
A = np.zeros([N_r, N_r])       
Atilde = np.zeros([N_r, N_r, 2, 2])          
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

    if count >= 2:
        d = int(D_cut**2)
    else:
        d = int(N_r**(2*(count+1)))

    M = contract_reshape(T, T, d)   
    M_prime = M.reshape(d, d*(N_r**2))     
    MMdag_prime = np.dot(M_prime, dagger(M_prime))  

    w, U = LA.eigh(MMdag_prime)   # Slow alternative : U, s1, V = LA.svd(MMdag_prime)
    idx = w.argsort()[::-1]
    s1 = w[idx]
    U = U[:,idx] 

    if np.size(U,1) > D_cut: 

        s = s1[:D_cut] 
        U = U[:,:D_cut]  
        eps += 1.0 - (sum(s)/sum(s1))


    else:
        s = s1
        eps = 1.0 - (sum(s)/sum(s1))

        

    count += 1 

    M_new = ncon((U, M),([1,-1], [1,-2,-3,-4])) 
    M_new = ncon((M_new, U),([-1,1,-3,-4], [1,-2]))  # With Dbond=41, this combination took 54 sec

    norm = LA.norm(M_new)
 
    nc += (2**((2*Niters)-count)) * np.log(norm)

    if norm != 0:   # and count != Niters
        T = M_new/norm

    else: 
      T = M_new
    
    return T, eps, nc, count  



def renyi(Rho, nmax): 

    for i in range (0,nmax):

        rho[i] = matrix_power(Rho, i+2)           # Raise to power "n" [\rho_A]^n 
        S[i] = math.log(np.trace(rho[i]))/(-1-i)  # Take log and divide by 1-n  
    
    return S 


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

    T = T.transpose(2,3,1,0)   
    T = np.einsum("iikl->kl", T)

    Tnew = np.einsum("ij,kl", T, T)   # Assuming system/sub-system of same size!
    # Need to try different sub-system sizes!
    Tnew = Tnew.reshape(D_cut**2, D_cut**2)


    for i in range (0, Niters):

        Tnew = np.matmul(Tnew,Tnew)              # 0. Raise over the time slices
        Tnew /= np.trace(Tnew)


    Tnew = Tnew/np.trace(Tnew)                   # 1. Normalize to recognize (T)^Nt as \rho
    Tnew = Tnew.reshape(D_cut,D_cut,D_cut,D_cut) # 2. Expose indices for A & B respectively 
    rho_A = np.einsum("klii->kl", Tnew)          # 3. Trace over environment/ subsystem B, or A as chosen
    rho_A = rho_A/np.trace(rho_A)                # 4. Normalize the density matrix again    

    EE = renyi(rho_A, nmax)
    print ("S_Renyi_2 =", EE[0])

    if verbose:
        print ("Finished",count,"C.G. steps with",D_cut,"states, " "kappa =", kappa, "and beta", beta) 

print ("-----------------------------------------------------------------")          
print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
