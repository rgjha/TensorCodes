# Tensor formulation of a classical statistical 2d model
# This proceeds by blocking simulatenously along both directions. 
# Reproduces free energy and magentization from arxiv:1309.04963

# Speedup 7x by feeding square matrix for SVD not rectangular : Dec 7, 2019 
# D=47 with Niter=30 takes ~  2.5 hours with SVD (The timing is on Symmetry machine at PI)
# Randomized SVD behaves badly for small 'h'! 

import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
from scipy import special
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix
from sklearn.utils.extmath import randomized_svd
import time
import datetime 
from ncon import ncon

if len(sys.argv) < 3:
  print("Usage:", str(sys.argv[0]), "<Temperature, T>  <h>" )
  sys.exit(1)

Temp =  float(sys.argv[1])
beta = float(1.0/Temp)
h =  float(sys.argv[2])

D=47
D_cut=47 
Niters=30
Ns = int(2**((Niters)))
Nt = Ns  
vol = Ns**2   
numlevels = Niters
norm_all = [0 for x in range(numlevels+1)]

if D%2 == 0:
    print ("D must be odd for now")
    sys.exit(1) 


Dn = int(D/2.0)

startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 


A = np.zeros([D])     
L = np.zeros([D])               


def tensorsvd(input,left,right,D):
    '''Reshape an input tensor into a rectangular matrix with first index corresponding
    to left set of indices and second index corresponding to right set of indices. Do SVD
    and then reshape U and V to tensors [left,D] x [D,right]  
    '''

    T = np.transpose(input,left+right)
    left_index_list = []
    for i in range(len(left)):
        left_index_list.append(T.shape[i])
    xsize = np.prod(left_index_list) 
    right_index_list = []
    for i in range(len(left),len(left)+len(right)):
        right_index_list.append(T.shape[i])
    ysize = np.prod(right_index_list)
    T = np.reshape(T,(xsize,ysize))
    
    
    U, s, V = np.linalg.svd(T,full_matrices = False)

    #X = sparse_random(100, 100, density=0.01, format='csr', random_state=42)
    #svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    #svd.fit(X)
    #U, s, V = randomized_svd(T, n_components=D+10, n_iter=10,random_state=None)
    
    if D < len(s):
        s = np.diag(s[:D])
        U = U[:,:D]
        V = V[:D,:]
    else:
        D = len(s)
        s = np.diag(s)

    U = np.reshape(U,left_index_list+[D])
    V = np.reshape(V,[D]+right_index_list)
        
        
    return U, s, V


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

def CG_step(matrix, in2):

    T = matrix  
    TI = in2 

    AAdag = ncon([T,T,T,T],[[-2,1,2,5],[-1,5,3,4],[-4,1,2,6],[-3,6,3,4]])
    U, s, V = tensorsvd(AAdag,[0,1],[2,3],D_cut) 
    A = ncon([U,T,T,U],[[1,2,-1],[2,-2,4,3],[1,3,5,-4],[5,4,-3]])
    B = ncon([U,TI,T,U],[[1,2,-1],[2,-2,4,3],[1,3,5,-4],[5,4,-3]])

    AAAAdag = ncon([A,A,A,A],[[1,-1,2,3],[2,-2,4,5],[1,-3,6,3],[6,-4,4,5]])
    U, s, V = tensorsvd(AAAAdag,[0,1],[2,3],D_cut)  
    AA = ncon([U,A,A,U],[[1,2,-2],[-1,1,3,4],[3,2,-3,5],[4,5,-4]])
    #BA2 = ncon([U,B,A,U],[[1,2,-2],[-1,1,4,3],[4,2,-3,5],[3,5,-4]]) 
    BA = ncon([U,B,A,U],[[1,2,-2],[-1,1,3,4],[3,2,-3,5],[4,5,-4]]) 
    #print ("Are they same? ", np.allclose(BA, BA2))
    # Small typo which makes no difference [see above]: 2 March, 2020

    # U, s, V = randomized_svd(T, n_components=D, n_iter=4,random_state=None)

    maxAA = np.max(AA)
    AA = AA/maxAA # Normalize by largest element of the tensor
    BA = BA/maxAA
        
    return AA, BA, maxAA


def get_tensor():

    for i in range (-Dn,Dn+1):
        L[i+Dn] = np.sqrt(sp.special.iv(i, beta))
 
    out = ncon((L, L, L, L),([-1],[-2],[-3],[-4])) 
    # Alt: T = np.einsum("i,j,k,l->ijkl", L, L, L, L)
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
 
    out = ncon((L, L, L, L),([-1],[-2],[-3],[-4])) 
    for l in range (-Dn,Dn+1):
        for r in range (-Dn,Dn+1):
            for u in range (-Dn,Dn+1):
                for d in range (-Dn,Dn+1):
                    index = l+u-r-d
                    out[l+Dn][r+Dn][u+Dn][d+Dn] *= 0.50 * (sp.special.iv(index-1, beta*h) + sp.special.iv(index+1, beta*h))

    return out 



if __name__ == "__main__":

 
    betah=beta*h
    T = get_tensor()
    Tim = get_site_mag()

    norm = LA.norm(T)
    T /= norm 
    Tim /= norm 


    Z = ncon([T,T,T,T],[[7,5,3,1],[3,6,7,2],[8,1,4,5],[4,2,8,6]])
    # See attached files for diagrams!
    #Z = ncon([T,T],[[1,-1,2,-2],[2,-3,1,-4]])
    #Z = ncon([Z,Z],[[1,2,3,4],[2,1,4,3]])
    C = 0
    N = 1
    C = np.log(norm)
    Free = -Temp*(np.log(Z)+4*C)/(4*N)

    for i in range (Niters):

        
        T, Tim, norm = CG_step(T, Tim)
        C = np.log(norm)+4*C
        N *= 4.0
        Free = -Temp*(np.log(Z)+4*C)/(4*N)
        print ("Free energy -> ", Free) 
        if i == Niters-1:

            
            #NUM = (np.einsum('ruru',T))**2
            #Y1 = NUM/np.einsum('rulu, ldrd',T, T)
            #X1 = NUM/np.einsum('ruld, ldru',T, T)     # Refs. 0903.1069 and 1706.03455
            #print ("FP value, ", X, Y)

            Z1 = ncon([T,T],[[1,-1,2,-2],[2,-3,1,-4]])
            Z = ncon([Z1,Z1],[[1,2,3,4],[2,1,4,3]])
            Free = -Temp*(np.log(Z)+4*C)/(4*N)

            P = ncon([Tim,T],[[1,-1,2,-2],[2,-3,1,-4]])
            P = ncon([P,Z1],[[1,2,3,4],[2,1,4,3]])

            r = (P/Z)
            print ("Mag. -> ", r)

               
    f=open("mag_data.txt", "a+")    
    f.write("%4.10f \t %4.10f \t %4.10f \t %2.0f \t %2.0f \t %4.10f \n" % (Temp, Free, r, Niters, D_cut, h)) 
    f.close()         
    #print (Temp,h,Free,r, X1, Y1)
    print (Temp,h,Free,r)
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
