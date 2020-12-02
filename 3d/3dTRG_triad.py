# Tensor formulation of 3d model using triad method 
# Free energy at T = 4.5115 is -3.51 
# Ref: https://arxiv.org/abs/1912.02414
# Now using "contract" which seems much faster than NCON
# https://doi.org/10.21105/joss.00753
# https://github.com/dgasmith/opt_einsum

import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
from scipy import special
from scipy.linalg import sqrtm
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
from matplotlib import pyplot as plt
import time
import datetime
from opt_einsum import contract


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 


if len(sys.argv) < 4:
  print("Usage:", str(sys.argv[0]), "<Niter, Dcut, System choice: 0 for Ising, 1 for U(1) model>")
  sys.exit(1)


Niter = int(sys.argv[1])
Dcut = int(sys.argv[2])
choice = int(sys.argv[3])


def dagger(a):

    return np.conjugate(np.transpose(a))


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

    U, s, V = sp.linalg.svd(T, full_matrices=False) 
    # Interchange between RSVD and usual SVD 
    #U, s, V = randomized_svd(T, n_components=D, n_iter=4,random_state=None)
    
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

def Z3d_Ising(beta):

    a = math.sqrt(np.cosh(beta))
    b = math.sqrt(np.sinh(beta)) 
    W = np.array([[a,b],[a,-b]])
    Id = np.eye(2)

    A = contract("ax, ay -> xya", W, W)
    B = contract("ab, az -> azb", Id, W)
    C = contract("bc, bz -> bzc", Id, W)
    D = contract("cy, cx -> cyx", W, W)

    return A, B, C, D


def Z3d_XY(beta, h, Dn):

    betah = beta*h 

    for i in range (-Dn,Dn+1):
        L[i+Dn] = np.sqrt(sp.special.iv(i, beta))

    out = contract("i,j,k,l,m,n->ijklmn", L, L, L, L, L, L)
    for l in range (-Dn,Dn+1):
        for r in range (-Dn,Dn+1):
            for u in range (-Dn,Dn+1):
                for d in range (-Dn,Dn+1):
                    for f in range (-Dn,Dn+1):
                        for b in range (-Dn,Dn+1):

                            index = l+u+f-r-d-b
                            out[l+Dn][r+Dn][u+Dn][d+Dn][f+Dn][b+Dn] *= sp.special.iv(index, betah)

    return out


def Z3d_U1(beta, D):

    A = np.zeros([D, D]) 

    for i in range (D):
        for j in range (D):
            A[i][j] = sp.special.iv(i-j, beta)

    W = LA.cholesky(A) 
    #out = contract("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W)

    Id = np.eye(D)
    A = contract("ax, ay -> xya", W, W)
    B = contract("ab, az -> azb", Id, W)
    C = contract("bc, bz -> bzc", Id, W)
    D = contract("cy, cx -> cyx", W, W)

    return A,B,C,D

    
def coarse_graining(in1, in2, in3, in4,impure=False):

    A = in1
    B = in2
    C = in3
    D = in4 

    S1 = contract('xyd,iyj->xidj', A, np.conjugate(A))
    ap = np.shape(S1)[0] * np.shape(S1)[1]
    bp = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(ap,bp))

    S2 = contract('dze,izj->diej', B, np.conjugate(B))
    a = np.shape(S2)[0] * np.shape(S2)[1]
    b = np.shape(S2)[2] * np.shape(S2)[3]
    S2 = np.reshape(S2,(a,b))

    Tmp = contract('fyx,iyx->fi', D, np.conjugate(D))
    R2 = contract('ewf,ijk,fk->eiwj', C, np.conjugate(C), Tmp)
    a = np.shape(R2)[0] * np.shape(R2)[1]
    b = np.shape(R2)[2] * np.shape(R2)[3]
    R2mat = np.reshape(R2,(a,b))

    Tmp = contract('bizz->bi', R2)
    R3 = contract('awb,ijk,bk->aiwj', B, np.conjugate(B), Tmp)
    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))

    Kprime = S1 @ S2 @ R2mat @ R3mat.T @ S1.T
    a = int(np.sqrt(np.shape(Kprime)[0]))
    b = int(np.sqrt(np.shape(Kprime)[1]))

    K = np.reshape(Kprime,(b,a,b,a))  # K_x1,x2,x3,x4         
    U, s1, UL = tensorsvd(K,[0,2],[1,3],int(Dcut)) 

    # Now finding "V"
    S1 = contract('ijk,ipq->jpkq', A, np.conjugate(A))
    ap = np.shape(S1)[0] * np.shape(S1)[1]
    bp = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(ap,bp))

    Tmp = contract('ijkk->ij', R2)
    R3 = contract('ijk,pqr,kr->ipjq', B, np.conjugate(B), Tmp)

    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))

    Kprime = S1 @ S2 @ R2mat @ R3mat.T @ S1.T
    a = int(np.sqrt(np.shape(Kprime)[0]))
    b = int(np.sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,a,b,a))
    V, s1, VL = tensorsvd(K,[0,2],[1,3],Dcut)

    # Free some arrays which are no longer needed  
    del K
    del Kprime
    del S1 

    # UC_abyxz
    # Note that there is typo in Eq. (17) of arXiv:1912.02414
    # Note that memory wise this is most expensive step. O(D^5) 
    # Also, DC below. 

    #UC = contract('azc,cqp,pix -> azqix',C,D,U)
    #UC = contract('azqix,bji,qjy -> abyxz',UC,D,V)

    Tmp1 = contract('cqp,pix -> cqix',D,U)
    Tmp2 = contract('bji,qjy -> biqy',D,V)
    Tmp3 = contract('cqix,biqy -> cxby',Tmp1,Tmp2)
    UC = contract('azc,cxby -> abyxz',C,Tmp3)

    #UC = np.tensordot(C,D,axes=([2,0])) # azqp 
    #UC = np.tensordot(UC,U,axes=([3,0])) # azqix 
    #UC = np.tensordot(UC,D,axes=([3,2])) # azqxbj 
    #UC = contract('azqxbj,qjy -> abyxz',UC,V)
    #UC = np.tensordot(UC,V,axes=([2,0],[5,1])) # azqxbj * qjy --> azxby --> 0,3,4,2,1
    
    # M_new = np.tensordot(M_new,U,axes=([1,0])) # UM_ajcd * U_jb --> UMU_acdb

    MC = contract('ijk,pjr->ipkr', B, C)
    Tmp = contract('ijkl,klabc->ijabc', MC, UC)

    del MC
    del UC
    G, st, D = tensorsvd(Tmp,[0,1,2],[3,4],Dcut) 
    G = contract('ijka,al->ijkl', G, st)  

    # DC = B_dzb * U*_pix * A_pqa * V*_qjy * A_ijd
    Tmp1 = contract('pix,pqa->ixqa', np.conjugate(U), A)
    Tmp2 = contract('qjy,ijd->qyid', np.conjugate(V), A)
    DC = contract('ixqa,qyid->xayd', Tmp1, Tmp2)
    DC = contract('dzb,xayd->zxyab', B, DC)
 
    Tmp2 = contract('ijkab,abmn->ijkmn', DC, G)
    A, st2, MCprime = tensorsvd(Tmp2,[0,1],[2,3,4],Dcut) 

    MCprime = contract('ij,jklm->iklm', st2, MCprime)
    B, st3, C = tensorsvd(MCprime,[0,1],[2,3],Dcut)

    # Split singular piece here...
    sing = sqrtm(st3) 
    B = contract('ijk,kp->ijp', B, sing)
    C = contract('kj,jip->kip', sing, C)

    return A,B,C,D 


if __name__ == "__main__":


    if choice == 0:
        temp = np.arange(4.5115, 4.5116, 0.0001).tolist()
        Nsteps = int(np.shape(temp)[0])
        f = np.zeros(Nsteps)

    if choice == 1:
        beta = np.arange(1.0, 1.1, 0.05).tolist()
        Nsteps = int(np.shape(beta)[0])
        f = np.zeros(Nsteps)

    for p in range (0, Nsteps):

        if choice == 0:
            A, B, C, D = Z3d_Ising(1.0/temp[p])
        if choice == 1:
            A, B, C, D = Z3d_U1(beta[p],Dcut)
    
        CU = 0.0 

        for iter in range (Niter):

            A, B, C, D = coarse_graining(A,B,C,D)  
            print ("Finished", iter+1, "of", Niter , "steps of CG")
            norm = np.max(A) * np.max(B) * np.max(C) * np.max(D) 
            div = np.sqrt(np.sqrt(norm))

            A  /= div
            B  /= div
            C  /= div
            D  /= div
            CU = CU + np.log(norm)/(2.0**(iter+1))

        
            if iter == Niter-1:

                #Z = contract('dfa,ahb,bic,cge,dfj,jhk,kim,mge', A, B, C, D, np.conjugate(A), np.conjugate(B), np.conjugate(C), D)  

                Tmp1 = contract('dfa,dfj->aj',A,np.conjugate(A))
                Tmp2 = contract('cge,mge->cm',D,np.conjugate(D))
                Tmp3 = contract('ahb,jhk->abjk',B,np.conjugate(B))
                Tmp4 = contract('aj,abjk->bk',Tmp1,Tmp3)
                Tmp5 = contract('bic,kim->bckm',C,np.conjugate(C))
                Z = contract('bckm,bk,cm',Tmp5,Tmp4,Tmp2)


                if choice == 0:  
                    Free = -(temp[p])*(CU + (np.log(Z)/(2.0**Niter)))
                    f[p] = Free 
                    print ("Free energy is ", round(Free,4), " @ T =", temp[p], "with bond dimension", Dcut)

                if choice == 1:  
                    Z_U1 = CU + (np.log(Z)/(2.0**Niter))
                    f[p] = -Z_U1
                    print (round(beta[p],4),round(-Z_U1,6))

    # Make plots if needed! 

    if choice == 0 and Nsteps > 3:

        dx = temp[1]-temp[0] # Assuming equal spacing ...
        dfdx = np.gradient(f, dx) 
        f = plt.figure()
        plt.plot(temp, dfdx, marker="*", color = "r")
        plt.grid(True)
        plt.title('3d classical Ising model using Triad TRG', fontsize=15)
        plt.xlabel('T', fontsize=13)
        plt.ylabel('df/dT', fontsize=13)
        plt.show()   
        f.savefig("plot1_ising.pdf", bbox_inches='tight')


    if choice == 1 and Nsteps > 3:

        dx = beta[1]-beta[0]
        dfdx = np.gradient(f, dx) 
        out = [] 
        for i in range(0, len(dfdx)): 
            out.append(dfdx[i] * beta[i] * (1.0/3.0)) 

        f = plt.figure()
        plt.plot(beta, out, marker="*", color = "r")
        plt.grid(True)
        plt.title('3d U(1) model using Triad TRG', fontsize=15)
        plt.xlabel(r'$\beta$')
        plt.ylabel('<S>', fontsize=13)
        plt.show()   
        f.savefig("plot1_U1.pdf", bbox_inches='tight')


    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
    

