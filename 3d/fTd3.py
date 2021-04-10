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
# First pass: April 8, 2021


startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 


if len(sys.argv) < 7:
  print("Usage:", str(sys.argv[0]), "<Niter, Niter_codim1, Dcut, start, end, incr")
  sys.exit(1)

Niter = int(sys.argv[1])
Niter2d = int(sys.argv[2])
Dcut = int(sys.argv[3])
start = float(sys.argv[4]) 
end = float(sys.argv[5])
incr = float(sys.argv[6])


if Niter%3 != 0:
  print("Niter need to be a multiple of 3")
  sys.exit(1)


def dagger(a):
    return np.conjugate(np.transpose(a))

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def tensorsvd(input,left,right,cut):
    
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
    if cut < len(s): 
        s = np.diag(s[:cut])
        U = U[:,:cut]
        V = V[:cut,:]
    else:
        cut = len(s)
        s = np.diag(s)

    U = np.reshape(U,left_index_list+[cut])
    V = np.reshape(V,[cut]+right_index_list)
        
    return U, s, V


def Z3d(beta, bdim):

    size = int(2*bdim + 1)
    A = np.zeros([size, size]) 

    for i in range (-bdim, bdim+1):
        for j in range (-bdim, bdim+1):
            A[i+bdim][j+bdim] = sp.special.iv(i-j, beta)

    W = LA.cholesky(A) 
    #out = contract("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W)

    Id = np.eye(size)
    A = contract("ax, ay -> xya", W, W)
    B = contract("ab, az -> azb", Id, W)
    C = contract("bc, bz -> bzc", Id, W)
    D = contract("cy, cx -> cyx", W, W)

    return A,B,C,D



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


def coarse_graining2d(matrix, cut):

    T = matrix  
    #AAdag = ncon([T,T,T,T],[[-2,1,2,5],[-1,5,3,4],[-4,1,2,6],[-3,6,3,4]])
    AAdag = contract('jabc,icde,labf,kfde->ijkl', T, T, T, T)
    U, s, V = tensorsvd(AAdag,[0,1],[2,3],cut) 
    #A = ncon([U,T,T,U],[[1,2,-1],[2,-2,4,3],[1,3,5,-4],[5,4,-3]])
    A = contract('abi,bjce,aedl,dck->ijkl', U, T, T, U)
    #AAAAdag = ncon([A,A,A,A],[[1,-1,2,3],[2,-2,4,5],[1,-3,6,3],[6,-4,4,5]])
    AAAAdag = contract('aibc,bjef,akgc,glef->ijkl', T, T, T, T)
    U, s, V = tensorsvd(AAAAdag,[0,1],[2,3],cut)  
    #AA = ncon([U,A,A,U],[[1,2,-2],[-1,1,3,4],[3,2,-3,5],[4,5,-4]])
    AA = contract('abj,iade,dbkc,ecl->ijkl', U, A, A, U)
    maxAA = np.max(AA)
    AA = AA/maxAA # Normalize by the largest element of the tensor
        
    return AA, maxAA



if __name__ == "__main__":


    beta = np.arange(start, end, incr).tolist()
    Nsteps = int(np.shape(beta)[0])
    f = np.zeros(Nsteps)
    mag = np.zeros(Nsteps)

    for p in range (0, Nsteps):

        A, B, C, D  = Z3d(beta[p], Dcut)
        CU = 0.0

        for iter in range (Niter):
            
            A, B, C, D = coarse_graining(A,B,C,D)   
            norm = LA.norm(A)*LA.norm(B)*LA.norm(C)*LA.norm(D) 
            div = np.sqrt(np.sqrt(norm))
            A  /= div
            B  /= div
            C  /= div
            D  /= div
            CU += np.log(norm)/(2.0**(iter+1))
            print ("# Three-dimensional iteration #", iter+1, "at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))       
            if iter == Niter-1:

                
                Tmp1 = contract('ika,dkj->iadj',A,D)
                Tmp2 = contract('amb,bcd->amcd',B,C)
                T2d = contract('iadj,amcd->ijcm',Tmp1,Tmp2) 


                #if Niter2d == 0:

                Tmp1 = contract('dfa,dfj->aj',A,np.conjugate(A))
                Tmp2 = contract('cge,mge->cm',D,np.conjugate(D))
                Tmp3 = contract('ahb,jhk->abjk',B,np.conjugate(B))
                Tmp4 = contract('aj,abjk->bk',Tmp1,Tmp3)
                Tmp5 = contract('bic,kim->bckm',C,np.conjugate(C))
                Z = contract('bckm,bk,cm',Tmp5,Tmp4,Tmp2)
                Zpar = CU + (np.log(Z)/(2.0**Niter))
                f[p] = -Zpar
                Free = f[p]*(1.0/beta[p])
                print (beta[p], Free)

                #else: 
                #CU = 0.0 
                for i in range (Niter2d):

                    T2d, norm2d = coarse_graining2d(T2d, Dcut)
                    CU += np.log(norm2d)/(4**(i+1))

                    if i == Niter2d-1:

                        #Z1 = ncon([T,T],[[1,-1,2,-2],[2,-3,1,-4]])
                        Z1 = contract('aibj,bkal->ijkl',T2d,T2d)
                        #Z = ncon([Z1,Z1],[[1,2,3,4],[2,1,4,3]])
                        Z = contract('abcd,badc',Z1,Z1)
                        Free1 = CU + (np.log(Z)/(4**(Niter2d)))
                        Free1 = -(1.0/beta[p])*Free1
                        print (round(beta[p],5), Free1)
                        print (round(beta[p],5), Free+Free1)


        file=open("compute_obs.txt", "a+")
        file.write("%4.8f \t %2.10e \t %2.0f \t %2.0f \t %2.0f \n" % (beta[p], Free, Niter, Niter2d, Dcut))
        file.close()


    endTime = time.time()
    print ("Running time:", round(endTime - startTime, 2),  "seconds")
    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
