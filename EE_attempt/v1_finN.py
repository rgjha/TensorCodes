# Started: 11 October 2019 

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
from packages import ncon

if len(sys.argv) < 3:
  print("Usage:", str(sys.argv[0]), "<rep.max> " "beta")
  sys.exit(1)

startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
print ("-----------------------------------------------------------------")

rmax=float(sys.argv[1])  
beta=float(sys.argv[2])  

if rmax > 1.0:
  print("<rep.max> can be maximum equal to 1 for now")
  sys.exit(1)

#****************************************
representation = [x for x in range (0, int(2.0*rmax)+1, 1)]  
dim = [x+1 for x in representation]               
N_r = int(sum(np.square(dim))) 
N_m = int(max(dim))
N = 4                                    
g_sq = (2*N)/beta         # beta/N = 2/g^2 
Niters = 5
Ns = int(2**((Niters)))
Nt = int(2**((Niters)))                              
vol = Ns*Nt
D_cut = 49  # D_cut=49 takes under a minute
B = np.zeros([N_r, N_r, N_r, N_r])
Imatrix = np.zeros([N, N])

#****************************************
def index(a, b, c):

    return int((a * (a + 1) * ((2.0*a) + 1)/(6.0)) + (a+2)*(a/2.0) + (a+1)*b + c)
    # sum_{j=0}^{j=N} (j+1)^2 = (N+1)*(N+2)*(2*N+3)/(6.0) is used. 
#****************************************

#****************************************
def det(beta, rep, NCOL):

    var = beta/NCOL
    for i in range (0, int(NCOL)):
        for j in range (0, int(NCOL)):

            ind = i - j + 2*rep 
            Imatrix[i][j] = 2.0 * ((2*rep)+1) * sp.special.iv(ind, var)/var

    F = np.linalg.det(Imatrix)  # det_{l+i-j+N}(beta/N) ~ F_r
    return F 
#****************************************

#****************************************
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
                B[j][k][l][m] = det(beta, 0.0, N)    # 1st element
                
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
                            B[j][k][l][m] = det(beta, r/2.0, N)  # Since r is twice as defined

    return  B
#****************************************

#****************************************
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

    else:
        s = s1

    count += 1 

    U = U.reshape(int(sqrt(np.size(U,0))),int(sqrt(np.size(U,0))),np.size(U,1))
    M_new =  ncon((U, T, T, U),([1,2,-1], [1,3,-3,4], [2,5,4,-4], [3,5,-2]))

    norm = LA.norm(M_new) 
    nc += (2**((2*Niters)-count)) * np.log(norm) 

    if norm != 0: 
        T = M_new/norm 

    else:  
      T = M_new
    
    return T, eps, nc, count  
#****************************************


if __name__ == "__main__":
 
    B = make_tensorB(representation)  # Plaquette tensor 
    T = B 
    norm = LA.norm(T)
    T /= norm 

    count = 0.0 
    eps = 0.0  

    nc = (2**((2*Niters))) * np.log(norm)

    for i in range (0,Niters):
        T, eps, nc, count = coarse_graining(T, eps, nc, count)   

    T = T.transpose(2,3,1,0) 
    T = np.einsum("iikl->kl", T)

    for i in range (0, Niters):

        T = np.dot(T, T)
        T /= np.trace(T)
    
    trT = np.einsum("ii", T)
    lnZ = np.log(trT) + nc    

print ("Finished",count,"coarse graining steps keeping",D_cut,"states and beta", beta) 
print ("Free energy density =", (-lnZ/vol))
print ("-----------------------------------------------------------------")          
print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))