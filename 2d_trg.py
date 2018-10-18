# Tensor formulation of 2d model
# Started : October 1, 2018 

# This uses the tensor renormalization group (TRG)
# based on the method presented in 1201.1144 which uses HOSVD 
# higher order SVD (singular value decomposition). 

# For 'exact' and more details refer to initial commits. 

import math
from math import sqrt
import numpy as np
import tensorflow as tf
import scipy as sp                       # For Bessel function 
from sympy.physics.quantum.cg import CG  # For CG coefficients (for now, using the function below) 
from sympy import S
from scipy import special
import sympy as sym
from sympy import simplify
import time 

startTime = time.time()

representation = [0.0, 1.0]              # By definition : 2r to avoid 1/2 integers. 
dim = [x+1 for x in representation]      # 2r + 1 
rep_max = int(max(representation))                 
N_r = int(sum(np.square(dim))) 
N_m = int(max(dim))

kappa=0.10                               # Coupling 
N=2                                      # SU(2) 
g=sqrt(2)
beta = 2.0*N/(g**2)
rho = 1                                 # Length of Higgs field
Nt = 4 
Ns = 3                                  # Number of spatial sites (need to be > 2)
vol = Nt * Ns
norder = 6 


A = np.zeros([size, size])                
B = np.zeros([size, size, size, size])

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
def contract_reshape(A, B, dim):
    if dim < 0:
        raise ValueError("Dimension of matrix is negative !!! ")
        return 0


    dummy1 = np.einsum("abcd, bpqr->apcqdr", A, B) # Do dummy1_acdpqr = A_abcd * B_bpqr 
    out = dummy1.reshape(N_r, N_r, dim, dim)  

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
        for k in range(0, lim+1):

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
    else:
        return 2.0 * (a+1.0) * (sp.special.iv((a+1.0), b)/(b)) 
##############################


##############################
def make_tensorA(rep):
    for r_a in rep:
        for r_b in rep:

            m_a = []
            m_b = []

            if r_a + r_b == 2.0 and abs(r_a - r_b) == 0.0:  # r_a = 1, r_b = 1
                temp1 = [0.0, 2.0] 
            elif r_a == r_b == 0.0:
                temp1 = [0.0]                                               
            else:
                temp1 = [1.0]  

            if r_a == r_b == 0:
                m_al = m_ar = m_bl = m_br = 0
                k = int(5.0 * (r_a/2.0) + (2*m_al) + m_ar)
                l = int(5.0 * (r_b/2.0) + (2*m_bl) + m_br)

                for sigma in temp1: 
                    CG1 = CGC((r_a/2.0), m_al, (sigma/2.0), (m_bl - m_al), (r_b/2.0), m_bl)
                    CG2 = CGC((r_a/2.0), m_ar, (sigma/2.0), (m_br - m_ar), (r_b/2.0), m_br) 
                    A[k][l] += Fr((sigma), kappa) * CG1  * CG2 / (r_b + 1) 

            else:

                if r_a == 0:
                    m_a.append(0) 
                else:
                    for x in [-r_a, r_a]:
                       m_a.append(x/2.0)    

                if r_b == 0:
                    m_b.append(0) 
                else:
                    for x in [-r_b, r_b]:
                       m_b.append(x/2.0) 

            for m_al in m_a:
                for m_bl in m_b:

                    for m_ar in m_a:
                        for m_br in m_b:

                            k = int(5.0 * (r_a/2.0) + (2*m_al) + m_ar)
                            l = int(5.0 * (r_b/2.0) + (2*m_bl) + m_br)

                            for sigma in temp1:  
                                CG1 = CGC((r_a/2.0), m_al, (sigma/2.0), (m_bl - m_al), (r_b/2.0), m_bl)
                                CG2 = CGC((r_a/2.0), m_ar, (sigma/2.0), (m_bl - m_al), (r_b/2.0), m_br) 
                                A[k][l] += Fr((sigma), kappa) *  CG1  *  CG2 / (r_b + 1)
    return A  
##############################


##############################
def make_tensorB(rep):

    m_a = []
    m_b = []
    count = 0
    for r in rep:
        r_l = r_r = r_a = r_b = r 

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


def coarse_trg(matrix, first_time):

    # Singular values to keep in SVD --> norder 

    if first_time == 1 :


        norder = Nr**2 
        T = matrix 
        M2 = contract_reshape(T, T, N_r**2)
        M2 = M2.T   # To broadcast M_new = U * D2 * U properly.
        M = M2.reshape(N_r**2, N_r**4)       
        MMdag = np.matmul(M, dagger(M))   # MMdag = M' * (M')†
        U, s, V = LA.svd(MMdag) # Equation (11) of 1201.1144
        U_trun = U[:,:norder]  # Do not truncate the first one ! 
        M_new = np.einsum("ia, ijcd, jb->abcd", U_trun, M2, U_trun) # Equation (10) of 1201.1144
        M_new /= LA.norm(M_new) # M_abcd = U_ia * M_ijcd * U_jb 


    else :

        M_new = matrix 
        M2 = M_new
        M = M2.reshape(N_r**2, N_r**4)       
        MMdag = np.matmul(M, dagger(M))   # MMdag = M' * (M')† 
        U, s, V = LA.svd(MMdag) # Equation (11) of 1201.1144
        U_trun = U[:,:norder]  # Truncate ! 
        M_new = np.einsum("ia, ijcd, jb->abcd", U_trun, M2, U_trun)
        M_new /= LA.norm(M_new)


    return M_new


##############################


if __name__ == "__main__":
 
    A = make_tensorA(representation)  # Link tensor  
    B = make_tensorB(representation)  # Plaquette tensor

    # "einsum" is a very useful numpy tool for Einstein's summation convention 
    # Matrix multiplication --> np.einsum('ij,jk->ik', a, b)  # c_ik = a_ij * b_jk
    # Dot product --> np.einsum('i,i->', a, b)  # c = a_i * b_i 
    # Outer product --> np.einsum('i,j->ij', a, b)  # c_ij = a_i * b_j  
    # Transpose --> np.einsum('ij->ji', a)  # c_ji = a_ij

    # See http://ajcr.net/Basic-guide-to-einsum/ for more details 

    C = np.einsum("ip, pjkl->ijkl", A, B) # Do C_ijkl = A_ip * B_pjkl  
    T = np.einsum("ijkl, lq->ijkq", C, A) # Do T_ijkq = C_ijkl * A_lq = A_ip * B_pjkl * A_lq 

    # For ex, construction of T_ijkq = A_ip * B_pjkl * A_lq is as :

    #         |                     |                   |                         |          |                    |
    #         |                     | k                 |                         |          | k                  | 
    #         |                     |                   |                         |          |                    |
    #         |    i          p     |     j             |         goes to         |    i     |     j              |
    # ------  B -------- A -------  B ------- A ------- B -------   -->    ------ B -------- T  ------  A  ------ B ------- 
    #         |                     |                                             |          |
    #         |                     |                                             |          |
    #         |                     |  l                                          |          |
    #         A                     A                                             |          | q 
    #         |                     |                                             A          |
    #         |                     |                                             |
    #         |                     |  q                                          |
    #         |                     

    
    CGS1 = coarse_trg(T, 1)  # T, first-time 
    CGS2 = coarse_trg(CGS1, 0) 
 
    #print ("Shape of CGS2 is", np.shape(CGS2))


runningTime = (time.time()-startTime)
print "TOTAL RUNNING TIME = " % runningTime