# Tensor formulation of 3d model PCM using triad method 
import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
import itertools 
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


if len(sys.argv) < 5:
  print("Usage:", str(sys.argv[0]), "<Niter, Dcut, beta, r_max>")
  sys.exit(1)

Niter = int(sys.argv[1])
Dcut = int(sys.argv[2])
beta = float(sys.argv[3])
rmax = float(sys.argv[4])
rep = [x for x in range (0, int(2.0*rmax)+1, 1)]                  
N_r = int(sum(np.square([x+1 for x in rep]))) 
# N_r = 5, (14), (30), (55), 91, (140), 204, (285), (385), 506, 650, (819) 
Rrep = [] 

if rmax == 0:
    print ("Trivial!")
    sys.exit(1) 

if rmax == 0.5:
    A = np.zeros([N_r, N_r, 14])  
    B = np.zeros([14, N_r, 30]) 
    C = np.zeros([30, N_r, 55])
if rmax == 1.0:
    A = np.zeros([N_r, N_r, 55])  
    B = np.zeros([55, N_r, 140])
    C = np.zeros([140, N_r, 140])
if rmax == 1.5:
    A = np.zeros([N_r, N_r, 140])  
    B = np.zeros([140, N_r, 385])
    C = np.zeros([385, N_r, 385])
if rmax == 2.0:
    A = np.zeros([N_r, N_r, 285])  
    B = np.zeros([285, N_r, 819])
    C = np.zeros([819, N_r, 819])



def dagger(a):

    return np.conjugate(np.transpose(a))


def index(a, b, c):

    return int((a)*((a) + 1)*((2.0*a) + 1)/(6.0) + (a+2)*(a/2.0) + (a+1)*b + c)
    # sum_{j=0}^{j=N} (j+1)^2 = (N+1)*(N+2)*(2*N+3)/(6.0) is used. 
    # Note: f[N_] := 1/6 (1+Floor[2 N]) (2+Floor[2 N]) (3+2 Floor[2 N]); 


def factorial(N):
    if N < 0:
        raise ValueError("N is negative !!! ", N)
        return 9999
    if math.floor(N) != N:
        raise ValueError("N must be an exact integer !!! ", N)
        return 9999 
    if N+1 == N:
        raise OverflowError("N is too large !!!")
    result = 1
    factor = 2
    while factor <= N:
        result *= factor
        factor += 1
    return result


def Fr(a, b):
    if b < 0 or a < 0:
        raise ValueError(" a or b is negative !!! ")
        return 0
    elif b==0 and a==1:
        return 0
    elif b==0 and a==0:
        return 2.0 * (a+1.0) * 0.50   # lim besselj[1,x]/x as x->0 = 0.5
    else:
        return 2.0 * (a+1.0) * (sp.special.iv((a+1.0), b)/(b)) 


# Returns SU(2) Clebsch-Gordon coefficients 
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


def Z3d_PCM(beta, D):

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


    S2 = contract('dze,izj->diej', B, np.conjugate(B))
    a = np.shape(S2)[0] * np.shape(S2)[1]
    b = np.shape(S2)[2] * np.shape(S2)[3]
    S2 = np.reshape(S2,(a,b))
    Tmp = contract('fyx,iyx->fi', D, np.conjugate(D))
    R2 = contract('ewf,ijk,fk->eiwj', C, np.conjugate(C), Tmp)
    a = np.shape(R2)[0] * np.shape(R2)[1]
    b = np.shape(R2)[2] * np.shape(R2)[3]
    R2mat = np.reshape(R2,(a,b))

    S1 = contract('xyd,iyj->xidj', A, np.conjugate(A))
    a = np.shape(S1)[0] * np.shape(S1)[1]
    b = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(a,b))
    Tmp = contract('bizz->bi', R2)
    R3 = contract('awb,ijk,bk->aiwj', B, np.conjugate(B), Tmp)
    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))

    #Kprime = S1 @ S2 @ R2mat @ R3mat.T @ S1.T
    Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)
    # Surprisingly, the above step is prone to some dependence on 
    # whethere we use 'matmul', 'dot', '@' and 'contract' 

    a = int(np.sqrt(np.shape(Kprime)[0]))
    b = int(np.sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,a,b,a))  # K_x1,x2,x3,x4         
    U, s1, UL = tensorsvd(K,[0,2],[1,3],int(Dcut)) 


    # Now finding "V"
    S1 = contract('ijk,ipq->jpkq', A, np.conjugate(A))
    a = np.shape(S1)[0] * np.shape(S1)[1]
    b = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(a,b))
    R3 = contract('ijk,pqr,kr->ipjq', B, np.conjugate(B), Tmp) # Use 'Tmp' from above
    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))


    Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)
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

    Tmp1 = contract('cqp,pix -> cqix',D,U)
    Tmp2 = contract('bji,qjy -> biqy',D,V)
    Tmp3 = contract('cqix,biqy -> cxby',Tmp1,Tmp2)
    
    #UC = contract('azc,cxby -> abyxz',C,Tmp3)
    MC = contract('ijk,pjr->ipkr', B, C)
    # Tmp = contract('ijkl,klabc->ijabc', MC, UC)
    Tmp = contract('ijab,azc,cxby->ijyxz', MC, C, Tmp3)

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

    # Split singular piece here!
    sing = sqrtm(st3) 
    B = contract('ijk,kp->ijp', B, sing)
    C = contract('kj,jip->kip', sing, C)

    return A,B,C,D 


def makeA(rep):

    m3 = []
    m1 = [] 
    n3 = [] 
    n1 = [] 
    M = [] 
    N = []

    for rp2, rp1 in itertools.product(rep, rep):
        for R in range(abs(rp2-rp1), abs(rp1+rp2)+1, 2):
        
            #print ("r+2, r+1, R", rp2/2.0, rp1/2.0, R/2.0)

            m3 = []
            n3 = [] 
            m1 = [] 
            n1 = [] 
            M = []
            N = [] 

            if rp2 == 0:
                m3.append(0.0)
                n3.append(0.0)
            else: 
                for x in range (-rp2, rp2+1, 2):
                    m3.append(x/2.0) if x/2.0 not in m3 else m3
                    n3.append(x/2.0) if x/2.0 not in n3 else n3

            if rp1 == 0:
                m1.append(0.0)
                n1.append(0.0)
            else: 
                for x in range (-rp1, rp1+1, 2):
                    m1.append(x/2.0) if x/2.0 not in m1 else m1
                    n1.append(x/2.0) if x/2.0 not in n1 else n1

            if R == 0:
                M.append(0.0)
                N.append(0.0)
            else: 
                for x in range (-R, R+1, 2):
                    M.append(x/2.0) if x/2.0 not in M else M
                    N.append(x/2.0) if x/2.0 not in N else N


            #print ("r_x+2, m3, n3", rp2/2.0, m3, n3)   
            #print ("r_x+1, m1, n1", rp1/2.0, m1, n1) 
            #print ("R, M, N", R/2.0, M, N) 
            #print ("r_x+2, r_x+1, R", rp2/2.0, rp1/2.0, R/2.0) 

            #print (m3, m1, M)

            for m3_e in m3:
                for n3_e in n3:
                    for m1_e in m1:
                        for n1_e in n1:
                            for M_e in M:
                                for N_e in N:

                                    i = index(rp2,m3_e,n3_e) 
                                    j = index(rp1,m1_e,n1_e) 
                                    k = index(R,M_e,N_e)

                                    #print (m3_e, n3_e, m1_e, n1_e, M_e, N_e)
                                    A[i][j][k] =  CGC((rp1/2.0), m1_e, (R/2.0), M_e, (rp2/2.0), m3_e) 
                                    A[i][j][k] *= CGC((rp1/2.0), n1_e, (rp2/2.0), n3_e, (R/2.0), N_e) 
                                    A[i][j][k] *= np.sqrt(Fr((rp1/2.0), beta) * Fr((rp1/2.0), beta)) 

    return  A


def makeB(rep):

    m3 = []
    m1 = [] 
    n3 = [] 
    n1 = [] 
    M = [] 
    N = []
    Rrep = [] 

    for rp2, rp1 in itertools.product(rep, rep):
        for R in range(abs(rp2-rp1), abs(rp1+rp2)+1, 2):
            if R not in Rrep:
                Rrep.append(R)

    #print ("Rrep", Rrep)
    #print ("Normal rep", rep)

    for R, rp3 in itertools.product(Rrep, rep):

        for Rprime in range(abs(R-rp3), abs(R+rp3)+1, 2):
            #print ("R, r+3, Rprime", R/2.0, rp3/2.0, Rprime/2.0)

            M = [] 
            N = []
            m5 = []
            n5 = [] 
            Mprime = []
            Nprime = [] 

            if R == 0:
                M.append(0.0)
                N.append(0.0)
            else: 
                for x in range (-R, R+1, 2):
                    M.append(x/2.0) if x/2.0 not in M else M
                    N.append(x/2.0) if x/2.0 not in N else N

            if rp3 == 0:
                m5.append(0.0)
                n5.append(0.0)
            else: 
                for x in range (-rp3, rp3+1, 2):
                    m5.append(x/2.0) if x/2.0 not in m5 else m5
                    n5.append(x/2.0) if x/2.0 not in n5 else n5

            if Rprime == 0:
                Mprime.append(0.0)
                Nprime.append(0.0)
            else: 
                for x in range (-Rprime, Rprime+1, 2):
                    Mprime.append(x/2.0) if x/2.0 not in Mprime else Mprime
                    Nprime.append(x/2.0) if x/2.0 not in Nprime else Nprime


            #kmax = 0.0
            #imax = 0.0 


            for M_e in M:
                for N_e in N:
                    for m5_e in m5:
                        for n5_e in n5:
                            for Mprime_e in Mprime:
                                for Nprime_e in Nprime:

                                    i = index(R,M_e,N_e) 
                                    j = index(rp3,m5_e,n5_e) 
                                    k = index(Rprime,Mprime_e,Nprime_e)

                                    #print ("Index", i, j, k)

                                    #if kmax < k: 
                                    #    kmax = k 
                                    #if imax < i: 
                                    #    imax = i

                                    #print ((R/2.0), M_e, (rp3/2.0), m5_e, (Rprime/2.0), Mprime_e)
                                    B[i][j][k] =  CGC((R/2.0), M_e, (rp3/2.0), m5_e, (Rprime/2.0), Mprime_e) 
                                    B[i][j][k] *= CGC((R/2.0), N_e, (rp3/2.0), n5_e, (Rprime/2.0), Nprime_e) 
                                    B[i][j][k] *= Fr((rp3/2.0), beta)
                                    B[i][j][k] /= np.sqrt(Rprime+1.0) 
                                    #B[i][j][k] *= np.sqrt(Fr((rp3/2.0), beta))*np.sqrt(Fr((rp3/2.0), beta))
                                    #B[i][j][k] = 0.0 
                                    #print (imax+1, kmax+1) 


    return  B,Rrep


def makeC(rep, Rrep):

    m3 = []
    m1 = [] 
    n3 = [] 
    n1 = [] 
    M = [] 
    N = [] 
    Rprimerep = []

    '''
    for rp2, rp1 in itertools.product(rep, rep):
        for R in range(abs(rp2-rp1), abs(rp1+rp2)+1, 2):
            if R not in Rrep:
                Rrep.append(R)
                '''

    for R, rp3 in itertools.product(Rrep, rep):
        for Rprime in range(abs(R-rp3), abs(R+rp3)+1, 2):
            if Rprime not in Rprimerep:
                Rprimerep.append(Rprime)


    for Rprime, rm3 in itertools.product(Rprimerep, rep):
        for Rdoubleprime in range(abs(Rprime-rm3), abs(Rprime+rm3)+1, 2):
            
            Mprime = [] 
            Nprime = []
            m6 = []
            n6 = [] 
            Mdoubleprime = []
            Ndoubleprime = [] 

            if Rprime == 0:
                Mprime.append(0.0)
                Nprime.append(0.0)
            else: 
                for x in range (-Rprime, Rprime+1, 2):
                    Mprime.append(x/2.0) if x/2.0 not in Mprime else Mprime
                    Nprime.append(x/2.0) if x/2.0 not in Nprime else Nprime

            if rm3 == 0:
                m6.append(0.0)
                n6.append(0.0)
            else: 
                for x in range (-rm3, rm3+1, 2):
                    m6.append(x/2.0) if x/2.0 not in m6 else m6
                    n6.append(x/2.0) if x/2.0 not in n6 else n6

            if Rdoubleprime == 0:
                Mdoubleprime.append(0.0)
                Ndoubleprime.append(0.0)
            else: 
                for x in range (-Rdoubleprime, Rdoubleprime+1, 2):
                    Mdoubleprime.append(x/2.0) if x/2.0 not in Mdoubleprime else Mdoubleprime
                    Ndoubleprime.append(x/2.0) if x/2.0 not in Ndoubleprime else Ndoubleprime


            kmax = 0.0
            imax = 0.0 


            for Mprime_e in Mprime:
                for Nprime_e in Nprime:
                    for m6_e in m6:
                        for n6_e in n6:
                            for Mdoubleprime_e in Mdoubleprime:
                                for Ndoubleprime_e in Ndoubleprime:

                                    i = index(Rprime,Mprime_e,Nprime_e) 
                                    j = index(rm3,m6_e,n6_e) 
                                    k = index(Rdoubleprime,Mdoubleprime_e,Ndoubleprime_e)

                                    C[i][j][k] =  CGC((Rdoubleprime/2.0), Mdoubleprime_e, (rm3/2.0), m6_e, (Rprime/2.0), Nprime_e) 
                                    C[i][j][k] *= CGC((Rdoubleprime/2.0), Ndoubleprime_e, (rm3/2.0), n6_e, (Rprime/2.0), Mprime_e) 
                                    C[i][j][k] *= Fr((rm3/2.0), beta)
                                    C[i][j][k] /= np.sqrt(Rprime+1.0) 

    return  C


if __name__ == "__main__":
 

    A = makeA(rep)
    #print ("Norm of A", LA.norm(A))
    B, Rrep = makeB(rep)
    #print ("Norm of B", LA.norm(B))
    C = makeC(rep, Rrep)
    print ("Norm of C", LA.norm(C))
    #D = makeA(rep)


    '''
    CU = 0.0 
    for iter in range (Niter):

        A, B, C, D = coarse_graining(A,B,C,D)  
        #print ("Finished", iter+1, "of", Niter , "steps of CG")
        norm = np.max(A)*np.max(B)*np.max(C)*np.max(D) 
        div = np.sqrt(np.sqrt(norm))

        A  /= div
        B  /= div
        C  /= div
        D  /= div
        CU += np.log(norm)/(2.0**(iter+1))

        
        if iter == Niter-1:

            Tmp1 = contract('dfa,dfj->aj',A,np.conjugate(A))
            Tmp2 = contract('cge,mge->cm',D,np.conjugate(D))
            Tmp3 = contract('ahb,jhk->abjk',B,np.conjugate(B))
            Tmp4 = contract('aj,abjk->bk',Tmp1,Tmp3)
            Tmp5 = contract('bic,kim->bckm',C,np.conjugate(C))
            Z = contract('bckm,bk,cm',Tmp5,Tmp4,Tmp2)
            # Pattern: dfa,ahb,bic,cge,dfj,jhk,kim,mge


            if choice == 0:  
                Free = -(temp[p])*(CU + (np.log(Z)/(2.0**Niter)))
                print ("Free energy is ", round(Free,4), " @ T =", round(temp[p],4), "with bond dimension", Dcut)
  

    '''


    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
    

