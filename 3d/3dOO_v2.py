# This was used to produce some results in arXiv: 2105.08066
# without any magentic field. For h \neq 0, the construction 
# of initial tensor is little more work (and slow!). 
# In the simulation directory, this file was once named: 3dXY_v6.py 


import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
from itertools import product
from scipy import special
from scipy.special import iv
from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds, eigs
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
import time
import datetime
from opt_einsum import contract


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 


if len(sys.argv) < 8:
  print("Usage:", str(sys.argv[0]), "<Niter, Dcut, Dn, start, end, incr, mag.field")
  sys.exit(1)

Niter = int(sys.argv[1])
Dcut = int(sys.argv[2])
Dn = int(sys.argv[3])
start = float(sys.argv[4]) 
end = float(sys.argv[5])
incr = float(sys.argv[6])
h = float(sys.argv[7])
L = np.zeros([2*Dn + 1])
Dcut_triad = int(Dcut*2) 
size_in = int(2*Dn + 1) 

if Niter%3 != 0:
  print("Niter need to be a multiple of 3")
  sys.exit(1)

if h != 0:
    print ("Only zero h here!")
    sys.exit(1)


def dagger(a):
    return np.conjugate(np.transpose(a))

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def tensorsvd(input,left,right,D):
    
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

    #T = T.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
    T = np.nan_to_num(T)
    U, s, V = sp.linalg.svd(T, full_matrices=False) 
    #U, s, V = np.linalg.svd(T, full_matrices=False)
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


def Z3d(beta, Dn):

    A = np.zeros((size_in, size_in, size_in*2)) 
    B = np.zeros((size_in*2, size_in, size_in*3)) 
    C = np.zeros((size_in*3, size_in, size_in*2))
    D = np.zeros((size_in*2, size_in, size_in))


    for i,j,k in product(range(-Dn, Dn+1), range(-Dn, Dn+1), range(-2*Dn, 2*Dn+1)):
        if ((i+j-k) == 0):
            A[i+Dn,j+Dn,k+2*Dn] = np.sqrt(iv(i, beta) * iv(j, beta))

    for i,j,k in product(range(-2*Dn, 2*Dn+1), range(-Dn, Dn+1), range(-3*Dn, 3*Dn+1)):
        if ((i+j-k) == 0):
            B[i+2*Dn,j+Dn,k+3*Dn] = np.sqrt(iv(j, beta))

    for i,j,k in product(range(-3*Dn, 3*Dn+1), range(-Dn, Dn+1), range(-2*Dn, 2*Dn+1)):
        if ((-i+j+k) == 0):
            C[i+3*Dn,j+Dn,k+2*Dn] = np.sqrt(iv(j, beta))

    for i,j,k in product(range(-2*Dn, 2*Dn+1), range(-Dn, Dn+1), range(-Dn, Dn+1)):
        if ((-i+j+k) == 0):
            D[i+2*Dn,j+Dn,k+Dn] = np.sqrt(iv(j, beta) * iv(k, beta))

    
    return A, B, C, D



def coarse_graining(in1, in2, in3, in4):


    S2 = contract('dze,izj->diej', in2, np.conjugate(in2))
    a = np.shape(S2)[0] * np.shape(S2)[1]
    b = np.shape(S2)[2] * np.shape(S2)[3]
    S2 = np.reshape(S2,(a,b))
    Tmp = contract('fyx,iyx->fi', in4, np.conjugate(in4))
    R2 = contract('ewf,ijk,fk->eiwj', in3, np.conjugate(in3), Tmp)
    del Tmp
    a = np.shape(R2)[0] * np.shape(R2)[1]
    b = np.shape(R2)[2] * np.shape(R2)[3]
    R2mat = np.reshape(R2,(a,b))

    S1 = contract('xyd,iyj->xidj', in1, np.conjugate(in1))
    a = np.shape(S1)[0] * np.shape(S1)[1]
    b = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(a,b))
    Tmp = contract('bizz->bi', R2)
    R3 = contract('awb,ijk,bk->aiwj', in2, np.conjugate(in2), Tmp)
    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))

    Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)

    a = int(np.sqrt(np.shape(Kprime)[0]))
    b = int(np.sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,a,b,a))         
    U, s1, UL = tensorsvd(K,[0,2],[1,3],int(Dcut)) 

    S1 = contract('ijk,ipq->jpkq', in1, np.conjugate(in1))
    a = np.shape(S1)[0] * np.shape(S1)[1]
    b = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(a,b))
    R3 = contract('ijk,pqr,kr->ipjq', in2, np.conjugate(in2), Tmp) 
    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))

    Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)
    #Kprime = S1 @ dum1 @ R3mat.T @ S1.T # Use Tmp from above

    a = int(np.sqrt(np.shape(Kprime)[0]))
    b = int(np.sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,a,b,a))
    V, s1, VL = tensorsvd(K,[0,2],[1,3],Dcut)


    Tmp1 = contract('cqp,pix -> cqix',in4,U)
    Tmp2 = contract('bji,qjy -> biqy',in4,V)
    Tmp3 = contract('cqix,biqy -> cxby',Tmp1,Tmp2)
    MC = contract('ijk,pjr->ipkr', in2, in3)
    Tmp = contract('ijab,azc,cxby->ijyxz', MC, in3, Tmp3)
    G, st, out4 = tensorsvd(Tmp,[0,1,2],[3,4],Dcut) 
    G = contract('ijka,al->ijkl', G, st)  
    Tmp1 = contract('pix,pqa->ixqa', np.conjugate(U), in1)
    Tmp2 = contract('qjy,ijd->qyid', np.conjugate(V), in1)
    DC = contract('ixqa,qyid->xayd', Tmp1, Tmp2)
    DC = contract('dzb,xayd->zxyab', in2, DC) 
    Tmp1 = contract('ijkab,abmn->ijkmn', DC, G)
    out1, st2, MCprime = tensorsvd(Tmp1,[0,1],[2,3,4],Dcut) 
    MCprime = contract('ij,jklm->iklm', st2, MCprime)
    out2, st3, out3 = tensorsvd(MCprime,[0,1],[2,3],Dcut)
    sing = sqrtm(st3) 
    out2 = contract('ijk,kp->ijp', out2, sing)
    out3 = contract('kj,jip->kip', sing, out3)

    return out1,out2,out3,out4



if __name__ == "__main__":


    beta = np.arange(start, end, incr).tolist()
    Nsteps = int(np.shape(beta)[0])
    f = np.zeros(Nsteps)
    mag = np.zeros(Nsteps)

    for p in range (0, Nsteps):

        A, B, C, D  = Z3d(beta[p], Dn)
        CU = 0.0

        for iter in range (Niter):
            
            A, B, C, D = coarse_graining(A,B,C,D)   

            if (iter+1)%3 == 0 and iter == Niter - 1:

                Tmp4 = contract('ajb,bjg->ag',B,C)
                Tmp5 = contract('ag,gfi->afi',Tmp4,D)
                Tmp6 = contract('iba,afi->fb',A,Tmp5)
                Xmeas = (np.trace(Tmp6))**2
                Xmeas /= np.trace(np.matmul(Tmp6, Tmp6))
                # B_ajb * C_bjg * D_gfi -> Z_afi
                # A_iba * Z_afi = T_bf

            norm = LA.norm(A)*LA.norm(B)*LA.norm(C)*LA.norm(D) 
            div = np.sqrt(np.sqrt(norm))
            A  /= div
            B  /= div
            C  /= div
            D  /= div
            CU += np.log(norm)/(2.0**(iter+1))
            print ("#Iteration", iter+1, "at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))       
            if iter == Niter-1: 

                Tmp1 = contract('dfa,dfj->aj',A,np.conjugate(A))
                Tmp2 = contract('cge,mge->cm',D,np.conjugate(D))
                Tmp3 = contract('ahb,jhk->abjk',B,np.conjugate(B))
                Tmp4 = contract('aj,abjk->bk',Tmp1,Tmp3)
                del Tmp1, Tmp3
                Tmp5 = contract('bic,kim->bckm',C,np.conjugate(C))
                Z = contract('bckm,bk,cm',Tmp5,Tmp4,Tmp2)
                # Pattern: dfa,ahb,bic,cge,dfj,jhk,kim,mge

                Zpar = CU + (np.log(Z)/(2.0**Niter))
                f[p] = -Zpar
                Free = f[p]*(1.0/beta[p])
                print (beta[p], Free) 
    

        file=open("computeX.txt", "a+")
        file.write("%4.8f \t %2.10e \t  %2.10e \t %2.0f \t %2.0f \t %2.0f \n" % (beta[p], Free, Xmeas, Niter, Dcut, Dn))
        file.close()

    endTime = time.time()
    print ("Running time:", round(endTime - startTime, 2),  "seconds")
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
    
