# Tensor formulation of 2d model
# Started : October 1, 2018 

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

representation = [0.0, 1.0]              # By definition : 2r to avoid 1/2 integers. 
dim = [1.0, 2.0]                         # 2r + 1 
size = int((dim[0]* dim[0]) + (dim[1]* dim[1]))   
kappa=0.10                               # Coupling 
N=2                                      # SU(2) 
g=sqrt(2)
beta = 2.0*N/(g**2)
rho = 1                                  # Length of Higgs field
A = np.zeros([size, size])                
B = np.zeros([size, size, size, size])
#B1 = tf.zeros([size, size, size, size])
#B1 = tf.Variable(tf.zeros([size, size, size, size]))  # Some experiments with tensor flow. TODO !

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)

'''Note that for rep=0 (r=0) , m can only take value = 0 ; for rep=1 (r=1/2), m = -1/2 or 1/2 
# Number of distinct values for m = dimension = rep + 1 ; rep = 1, 2
# We can compress three indices r_a, m_al, m_ar into : 5r_a + 2m_al + m_ar 
# such that it goes from 0...4 ; 0 when r=m1=m2=0  and 4 when r=1/2, m1=m2=1/2 
# Would be better to generalize it for r beyond 1/2 [TODO] !'''
 

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

if __name__ == "__main__":
 
    A = make_tensorA(representation)  # Link tensor  
    B = make_tensorB(representation)  # Plaquette tensor


    # "einsum" is a very useful numpy tool for Einstein's summation convention 
    # See http://ajcr.net/Basic-guide-to-einsum/ for more details 
    C = np.einsum("ip, pjkl->ijkl", A, B) # Do C_ijkl = A_ip * B_pjkl  
    D = np.einsum("ijkl, lq->ijkq", C, A) # Do D_ijkq = C_ijkl * A_lq = A_ip * B_pjkl * A_lq 
    D2 = np.einsum("abcd, bpqr->acdpqr", D, D) # Do D2_acdpqr = D_abcd * D_bpqr 


    # For ex, construction of D is as :

    #         |                     |                   |                         |          |
    #         |                     | k                 |                         |          | k 
    #         |                     |                   |                         |          |
    #         |    i          p     |     j             |                         |    i     |       j 
    #         B -------- A -------  B ------- A ------- B ----------      ---->   B -------- D  ------  A ------ B 
    #         |                     |                   |                         |          |
    #         |                     |                   |                         |          |
    #         |                     |  l                |                         |          |
    #         A                     A                   A                         |          | q 
    #         |                     |                   |
    #         |                     |                   |
    #         |                     |  q                |
    #         |                     |                   |

