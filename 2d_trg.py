# Tensor formulation of 2d model
# Started : October 1, 2018 

# This code uses modified version of the tensor renormalization 
# group (TRG) based on the method presented in 1201.1144 which uses HOSVD 
# higher order SVD (singular value decomposition). 

# For 'exact' and more details refer to initial commits. 

# Coarse grained tensor renormalization group (TRG) method
# was introduced in 0611687 by Levin and Nave. 

# Then in 1201.1144, by using the higher-order SVD, improvements 
# were made. Note that these TRG methods fail at criticality because 
# there still exists short-ranged correlations.

# In 2015, PRL 115, 180405 (arxiv : 1412.0732) introduced
# a new method called Tensor Network Renormalization (TNR)
# which performs better than TRG. With bond-dimension
# which is called D_cut in this code, TNR improves 
# exponentially compared to TRG which is polynomial. 
# Note that cost for both scales as O(D_cut**(2d-1)) i.e 
# O(D_cut**7) here. 

# Another good reference is 1510.03333 with some details 

# https://arxiv.org/abs/1801.04183 discusses the tensor network 
# formulation for two-dimensional lattice N=1 Wess-Zumino model 

import sys
import math
from math import sqrt
import numpy as np
import scipy as sp                       # For Bessel function 
from scipy import special
from numpy import linalg as LA
from numpy import ndarray
import time 
import datetime 

if len(sys.argv) < 2:
  print("Usage:", str(sys.argv[0]), "<Verbose or not>")
  sys.exit(1)


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
print ("-----------------------------------------------------------------")

representation = [0.0, 1.0]              # By definition : 2r 
dim = [x+1 for x in representation]      # 2r + 1
rep_max = int(max(representation))                 
N_r = int(sum(np.square(dim))) 
N_m = int(max(dim))

kappa=0.10                               # Coupling 
N=2                                      # SU(2) 
g=sqrt(2)
beta = 2.0*N/(g**2)
rho = 1                                  # Length of Higgs field
Niters = 6                             
Ns = int(2**((Niters)))  
Nt = Ns      
vol = Nt * Ns
D_cut = 40

# Time ~ 17 sec with D_cut=40 and Niters=6
# Time ~ 25 sec with D_cut=40 and Niters=8
# Time ~ 37 sec with D_cut=40 and Niters=10

if D_cut <= N_r**2:
  print("Usage: D_cut must be greater than N_r**2 for now")
  sys.exit(1)


verbose = int(sys.argv[1]) 


A = np.zeros([N_r, N_r])                
B = np.zeros([N_r, N_r, N_r, N_r])

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)


##############################
def index(a, b, c):

    return int((a)*((a) + 1)*((2.0*a) + 1)/(6.0) + (a+2)*(a/2.0) + (a+1)*b + c)
    # sum_{j=0}^{j=N} (j+1)^2 = (N+1)*(N+2)*(2*N+3)/(6.0) is used. 
##############################


##############################
def dagger(a):

    return np.transpose(a).conj() 
##############################
 

##############################
def factorial(N):
    if N < 0:
        raise ValueError("N is negative !!! ")
        return 9999
    if math.floor(N) != N:
        raise ValueError("N must be an exact integer !!! ")
        return 9999 
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
    if b <= 0 or a < 0:
        raise ValueError("Negative arguments!!! ")
        return 0
    elif b==0 and a==1:
        return 0
    elif b==0 and a==0:
        return 2.0 * (a+1.0) * 0.50   # lim besselj[1,x]/x as x->0 = 0.5
    else:
        return 2.0 * (a+1.0) * (sp.special.iv((a+1.0), b)/(b)) 
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
                j = int(5.0 * (r/2.0) + (2*m_al) + m_lb)
                k = int(5.0 * (r/2.0) + (2*m_ra) + m_br)
                l = int(5.0 * (r/2.0) + (2*m_al) + m_ra)
                m = int(5.0 * (r/2.0) + (2*m_lb) + m_br)
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

                            j = int(5.0 * (r/2.0) + (2*m_al) + m_bl)
                            k = int(5.0 * (r/2.0) + (2*m_ar) + m_br)
                            l = int(5.0 * (r/2.0) + (2*m_al) + m_ar)
                            m = int(5.0 * (r/2.0) + (2*m_bl) + m_br)
                            B[j][k][l][m] = Fr(r, beta)

    return  B
##############################



##############################
def make_Atilde(rep):

    for r_l in rep:
        for r_r in rep:

            m_l = []
            m_r = []

            ALPHA = [-0.5, 0.5]   # Rep chosen for ploop 
            BETA = [-0.5, 0.5] 


            if r_l == 0:
                m_l.append(0) 
            else:
                for x in [-r_l, r_l]:
                    m_l.append(x/2.0)  

            if r_r == 0:
                m_r.append(0) 
                    
            else:
                for x in [-r_r, r_r]:
                    m_r.append(x/2.0)  


            for m_la in m_l:
                for m_lb in m_l:
                    for m_ra in m_r:              
                        for m_rb in m_r:
                            for alpha in ALPHA:
                                for beta in BETA:

                                    k = index(r_l, m_la, m_lb)
                                    l = index(r_r, m_ra, m_rb)  
                                    al = int(((2*alpha)+1.0)/2.0) 
                                    be = int(((2*beta)+1.0)/2.0) 
                        
                                    val = 0 
                        
                                    for r in range(abs(r_l-1), abs(r_l+1)+1, 2):
                                        for sigma in range(abs(r_r-r), abs(r_r+r)+1, 2): 
                                        
                                            CG  = CGC(r/2.0, alpha+m_lb, sigma/2.0, m_ra - (beta+m_la), r_r/2.0, m_rb)
                                            CG *= CGC(r/2.0, beta+m_la, sigma/2.0, m_ra - (beta+m_la), r_r/2.0, m_ra)
                                            CG *= CGC(r_l/2.0, m_lb, (1/2.0), alpha, (r/2.0), (alpha + m_lb))
                                            CG *= CGC(r_l/2.0, m_la, (1/2.0), beta, (r/2.0), (beta + m_la))
                                            val += Fr(sigma, kappa) * CG / (r_r + 1)  

                                    Atilde[k][l][al][be] = val 
                                    
    return Atilde   
##############################



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
    
    w, U = LA.eigh(MMdag_prime)   # U, s1, V = LA.svd(MMdag_prime)
    idx = w.argsort()[::-1]
    s1 = w[idx]
    U = U[:,idx]

    if np.size(U,1) > D_cut: 

        s = s1[:D_cut] 
        U = U[:,:D_cut] 
        eps += 1.0 - (sum(s)/sum(s1))

    count += 1 

    M_new = np.tensordot(U,M,axes=([0,0])) # U_ia * M_ijcd = UM_ajcd 
    M_new = np.tensordot(M_new,U,axes=([1,0])) # UM_ajcd * U_jb --> UMU_acdb 
    M_new = np.transpose(M_new, (0,3,1,2))  # UMU_acdb --> UMU_abcd
    norm = LA.norm(M_new)

    nc += (2**((2*Niters)-count)) * np.log(norm) 

    if norm != 0:
        T = M_new/norm 

    else:
      print ("Norm of tensor is = ", norm)  
      sys.exit(1)

    return T, eps, nc, count


##############################



if __name__ == "__main__":
 
    A = make_tensorA(representation)  # Link tensor  
    B = make_tensorB(representation)  # Plaquette tensor
    Aprime = make_Atilde(representation) # Impure tensor for Polyakov loop

    # "einsum" is a very useful numpy tool for Einstein's summation convention 
    # Matrix multiplication --> np.einsum('ij,jk->ik', a, b)  # c_ik = a_ij * b_jk
    # Dot product --> np.einsum('i,i->', a, b)  # c = a_i * b_i 
    # Outer product --> np.einsum('i,j->ij', a, b)  # c_ij = a_i * b_j  
    # Transpose --> np.einsum('ij->ji', a)  # c_ji = a_ij

    # See http://ajcr.net/Basic-guide-to-einsum/ for more details 

    # But "einsum" is slow. We can use sequence of reshape, np.dot to contract 

    L = LA.cholesky(A) 
    Linverse = LA.inv(L)
    T = np.einsum("pjkl, pa, jb, kc, ld", B, L, L, L, L)

    # Unaware if tensor.dot can take > 2 tensors to contract ! TODO
    # Alternative : 
    #dum = np.tensordot(B,A,axes=([0,0]))   # BA_jkla 
    #dum = np.tensordot(dum,A,axes=([0,0])) # BAA_klab 
    #dum = np.tensordot(dum,A,axes=([0,0])) # BAAA_labc
    #T = np.tensordot(dum,A,axes=([0,0])) # BAAAA_abcd 

    T /= LA.norm(T)  

    # T is also sometimes called as "fundamental tensor". 
    # Indices are always written in the order : left, right, top, bottom, up, down 
    # For ex, construction of T_ijkq = A_ip * B_pjkl * A_lq is as :
    
    #               | c     
    #  -- A -- B -- A -- B -- A -- 
    #     |    |    | k  |    |         
    #     |  a |  p | j  |  b |     goes to 
    #  -- B -- A -- B -- A -- B--    --->    T_abcd 
    #     |    |    |    |    |         
    #     |    |    |l   |    |          
    #  -- A -- B -- A -- B -- A --
    #     |    |    |d   |    |  
    #     |    |    |    |    |
    #  -- B -- A -- B -- A -- B -- 
        
    
    count = 0.0 
    eps = 0.0  

    nc = (2**((2*Niters)-count)) * np.log(norm)  # First normalization

    for i in range (0,Niters):

        T, eps, nc, count = coarse_graining(T, eps, nc, count) 


    T = T.transpose(2,3,1,0)   

    dum = np.einsum("abcd,be->aecd", T, Linverse) 
    dum1 = np.einsum("if, ijkl->fjkl", Linverse, T)
    PL = np.einsum("aecd, efgh, fjkl->ajcgkdhl", dum, Aprime, dum1)    
    PL = np.einsum("iicgkdhl->cgkdhl", PL) 
    PL = PL.reshape(D_cut*D_cut*2,D_cut*D_cut*2) 
    T = np.einsum("iikl->kl", T)


    for i in range (0, Niters):

        PL = np.dot(PL, PL)
        T = np.dot(T, T)
        PL /= np.trace(T)
        T /= np.trace(T)


    trT = np.einsum("ii", T)
    PL = np.einsum("ii", PL)/trT
    PL = (Ploop/2.0)**(1/Nt)
    lnZ = np.log(trT) + nc            # Accumulated factors over Niters, note that usually np.log(trT) <<< nc
            

print ("Finished",count,"coarse graining steps keeping ",D_cut,"states")
print ("Free energy density =", (-lnZ/vol))  # Matched on November 8 with Judah 
print ("Polyakov line =", PL)  # Matched on December 11 
print ("-----------------------------------------------------------------")           
print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

