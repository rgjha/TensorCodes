# Tensor formulation of 2d model
# Started : October 1, 2018 

# This uses the tensor renormalization group (TRG)
# based on the method presented in 1201.1144 which uses HOSVD 
# higher order SVD (singular value decomposition). 

# For 'exact' and more details refer to initial commits. 

# Coarse grained tensor renormalization group (TRG) method
# was introduced in 0611687 by Levin and Nave. 

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
print ("STARTED : " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 

representation = [0.0, 1.0]              # By definition : 2r to avoid 1/2 integers. 
dim = [x+1 for x in representation]      # 2r + 1 
rep_max = int(max(representation))                 
N_r = int(sum(np.square(dim))) 
N_m = int(max(dim))

kappa=0.10                               # Coupling 
N=2                                      # SU(2) 
g=sqrt(2)
beta = 2.0*N/(g**2)
rho = 1                                  # Length of Higgs field
Nlayers = 4                             
Nt = 4 
Ns = int(2**((Nlayers+2)/(2.0)))        # Number of spatial sites (need to be > 2)
vol = Nt * Ns
D_bond = 20


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


    dummy1 = np.einsum("abcd, efdh->aebfch", A, B) # Do dummy1_acdpqr = A_abcd * B_bpqr 
    out = dummy1.reshape(dim, dim, N_r, N_r)  

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
def start_coarse_graining(matrix, order):

    # Truncation determined by --> norder

    T = matrix 
    M2 = contract_reshape(T, T, N_r**2)  # 4-index object

    # First step 
    M_prime = M.reshape(N_r**2, N_r**4)      
    MMdag_prime = np.matmul(M_prime, dagger(M_prime))   # MMdag = M' * (M')†  
    U, s, V = LA.svd(MMdag_prime) # Equation (11) of 1201.1144
    # Do not truncate the first one ! 
    M_new = np.einsum("ia, ijcd, jb->abcd", U, M, U) # Equation (10) of 1201.1144
    M = M_new/LA.norm(M_new) # Reassign 


    # Second step aka first truncation step 
    M = contract_reshape(M, M, N_r**4)
    M_prime = M.reshape(N_r**4, N_r**6)
    MMdag_prime = np.matmul(M_prime, dagger(M_prime))
    U, s, V = LA.svd(MMdag_prime) # Equation (11) of 1201.1144
    U = U[:,:D_bond]   
    M_new = np.einsum("ia, ijcd, jb->abcd", U, M, U)  # 
    M = M_new/LA.norm(M_new)
    
    return M
##############################


##############################
def coarse_graining(matrix, D_bond):

    M = contract_reshape(matrix, matrix, D_bond**2) # --> (40,40,5,5) 
    M_prime = M.reshape(D_bond**2, (N_r**2)*(D_bond**2))
    MMdag_prime = np.matmul(M_prime, dagger(M_prime))   # MMdag = M' * (M')† 
    U, s, V = LA.svd(MMdag_prime) # Equation (11) of 1201.1144
    U = U[:,:D_bond]   
    M_new = np.einsum("ia, ijcd, jb->abcd", U, M, U)  #  
    M = M_new/LA.norm(M_new)

    return M 
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

    # But "einsum" is slow. We can use sequence of reshape, np.dot to contract 

    B1 = B.reshape(N_r, N_r**3)
    C = np.dot((A), B1)
    C = C.reshape(N_r, N_r, N_r, N_r) 
    # Slow Alternative -> C = np.einsum("ip, pjkl->ijkl", A, B) # Do C_ijkl = A_ip * B_pjkl 

    C1 = C.reshape(N_r**3, N_r)
    T = np.dot((C1), A)
    T = T.reshape(N_r, N_r, N_r, N_r)
    # Slow alternative : T = np.einsum("ijkl, lq->ijkq", C, A) # Do T_ijkq = C_ijkl * A_lq = A_ip * B_pjkl * A_lq)


    T /= LA.norm(T) 

    # T is also sometimes called as "fundamental tensor".
    # For ex, construction of T_ijkq = A_ip * B_pjkl * A_lq is as :
    # Indices are always written in the order : left, right, top, bottom, up, down 

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

    
    CGS = start_coarse_graining(T)  
        
    count = 2 # coarse_graining done twice before this 
    for i in range (0,Nlayers-2):
        CGS1 = coarse_graining(CGS, D_bond) 
        count = count+1
        CGS = CGS1 
            

print ("Finished",count,"coarse graining steps keeping ",D_bond,"states")
print ("COMPLETED : " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

