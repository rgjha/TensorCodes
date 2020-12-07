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
from matplotlib import pyplot as plt
import time
import datetime
from opt_einsum import contract
from ncon import ncon 
                     
startTime = time.time()
print ("STARTED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 


if len(sys.argv) < 4:
  print("Usage:", str(sys.argv[0]), "<Niter, Dcut, r_max>")
  sys.exit(1)

Niter = int(sys.argv[1])
Dcut = int(sys.argv[2])
rmax = float(sys.argv[3])
rep = [x for x in range (0, int(2.0*rmax)+1, 1)]       
rep1 = [x for x in range (0, int(4.0*rmax)+1, 1)]       
rep2 = [x for x in range (0, int(6.0*rmax)+1, 1)]      
N_r = int(sum(np.square([x+1 for x in rep]))) 
# N_r = 5, (14), (30), (55), 91, (140), 204, (285), (385), 506, 650, (819), 1015, 1240, 1496, 1785, 2109 
N_r_p = int(sum(np.square([x+1 for x in rep1])))
N_r_pp = int(sum(np.square([x+1 for x in rep2])))

Rrep = [] 
Rprimerep = []
Rdprimerep = []

if rmax == 0:
    print ("Trivial!")
    sys.exit(1) 


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

    U, s, V = np.linalg.svd(T, full_matrices=False) 
    
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

    Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)

    a = int(sqrt(np.shape(Kprime)[0]))
    b = int(sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,a,b,a))         
    U, s1, UL = tensorsvd(K,[0,2],[1,3],int(Dcut)) 

    S1 = contract('ijk,ipq->jpkq', A, np.conjugate(A))
    a = np.shape(S1)[0] * np.shape(S1)[1]
    b = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(a,b))
    R3 = contract('ijk,pqr,kr->ipjq', B, np.conjugate(B), Tmp)
    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))


    Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)
    a = int(sqrt(np.shape(Kprime)[0]))
    b = int(sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,a,b,a))
    V, s1, VL = tensorsvd(K,[0,2],[1,3],Dcut)

    del K
    del Kprime
    del S1 

    Tmp1 = contract('cqp,pix -> cqix',D,U)
    Tmp2 = contract('bji,qjy -> biqy',D,V)
    Tmp3 = contract('cqix,biqy -> cxby',Tmp1,Tmp2)
    
    MC = contract('ijk,pjr->ipkr', B, C)
    Tmp = contract('ijab,azc,cxby->ijyxz', MC, C, Tmp3)

    G, st, D = tensorsvd(Tmp,[0,1,2],[3,4],Dcut) 
    G = contract('ijka,al->ijkl', G, st)  

    Tmp1 = contract('pix,pqa->ixqa', np.conjugate(U), A)
    Tmp2 = contract('qjy,ijd->qyid', np.conjugate(V), A)
    DC = contract('abcd,cpaq->bdpq', Tmp1, Tmp2)
    #DC = ncon([np.conjugate(U),A,np.conjugate(V),A],[[1,2,-1],[1,3,-2],[3,4,-3],[2,4,-4]])
    DC = contract('dzb,xayd->zxyab', B, DC)

    Tmp2 = contract('ijkab,abmn->ijkmn', DC, G)
    A, st2, MCprime = tensorsvd(Tmp2,[0,1],[2,3,4],Dcut) 
    MCprime = contract('ij,jklm->iklm', st2, MCprime)
    B, st3, C = tensorsvd(MCprime,[0,1],[2,3],Dcut)


    sing = sqrtm(st3) 
    # Note that both st3 & sing are diagonal matrix 
    # Check using: np.count_nonzero(st3 - np.diag(np.diagonal(st3)))
    B = contract('ijk,kp->ijp', B, sing)
    C = contract('kj,jip->kip', sing, C)
    

    return A,B,C,D 


def makeA(rep, beta):

    A = np.zeros([N_r, N_r, N_r_p])


    for rp2, rp1 in itertools.product(rep, rep):
        for R in range(abs(rp2-rp1), abs(rp1+rp2)+1, 2):

            #print ("Reps for A,", rp2/2.0, rp1/2.0, R/2.0)

            m3 = []
            n3 = [] 
            m1 = [] 
            n1 = [] 
            M = [] 
            N = []
        
            for x in range (-rp2, rp2+1, 2):
                m3.append(x/2.0) if x/2.0 not in m3 else m3
                n3.append(x/2.0) if x/2.0 not in n3 else n3
 
            for x in range (-rp1, rp1+1, 2):
                m1.append(x/2.0) if x/2.0 not in m1 else m1
                n1.append(x/2.0) if x/2.0 not in n1 else n1

            for x in range (-R, R+1, 2):
                M.append(x/2.0) if x/2.0 not in M else M
                N.append(x/2.0) if x/2.0 not in N else N


            for m3_e, n3_e, m1_e, n1_e, M_e, N_e in itertools.product(m3, n3, m1, n1, M, N):

                i, j, k = index(rp2,m3_e,n3_e), index(rp1,m1_e,n1_e), index(R,M_e,N_e)  

                A[i][j][k] =  CGC((rp1/2.0), m1_e, (rp2/2.0), m3_e,(R/2.0), M_e) 
                A[i][j][k] *= CGC((rp1/2.0), n1_e, (rp2/2.0), n3_e, (R/2.0), N_e) 
                A[i][j][k] *= sqrt(Fr((rp1/2.0), beta) * Fr((rp2/2.0), beta))


    return  A


def makeB(rep, beta):

    B = np.zeros([N_r_p, N_r, N_r_pp])

    for rp2, rp1 in itertools.product(rep, rep):
        for R in range(abs(rp2-rp1), abs(rp1+rp2)+1, 2):
            if R not in Rrep:
                Rrep.append(R)


    for R, rp3 in itertools.product(Rrep, rep):
        for Rprime in range(abs(R-rp3), abs(R+rp3)+1, 2):


            #print ("Reps for B,", R/2.0, rp3/2.0, Rprime/2.0)

            M = [] 
            N = []
            m5 = []
            n5 = [] 
            Mprime = []
            Nprime = [] 

            for x in range (-R, R+1, 2):
                M.append(x/2.0) if x/2.0 not in M else M
                N.append(x/2.0) if x/2.0 not in N else N

            for x in range (-rp3, rp3+1, 2):
                m5.append(x/2.0) if x/2.0 not in m5 else m5
                n5.append(x/2.0) if x/2.0 not in n5 else n5

            for x in range (-Rprime, Rprime+1, 2):
                Mprime.append(x/2.0) if x/2.0 not in Mprime else Mprime
                Nprime.append(x/2.0) if x/2.0 not in Nprime else Nprime


            for M_e, N_e, m5_e, n5_e, Mprime_e, Nprime_e in itertools.product(M, N, m5, n5, Mprime, Nprime):



                i, j, k = index(R,M_e,N_e), index(rp3,m5_e,n5_e), index(Rprime,Mprime_e,Nprime_e) 

                b1 = CGC((R/2.0), M_e, (rp3/2.0), m5_e, (Rprime/2.0), Mprime_e) 
                b2 = CGC((R/2.0), N_e, (rp3/2.0), n5_e, (Rprime/2.0), Nprime_e)
                b3 = sqrt(Fr((rp3/2.0), beta))/sqrt(Rprime+1.0) 

                B[i][j][k] =  b1 * b2 * b3




    return  B


def makeC(rep, beta):

    C = np.zeros([N_r_pp, N_r, N_r_p])

    for rm1, rm2 in itertools.product(rep, rep):
        for Rdprime in range(abs(rm1-rm2), abs(rm1+rm2)+1, 2):
            if Rdprime not in Rdprimerep:
                Rdprimerep.append(Rdprime)


    for Rdprime, rm3 in itertools.product(Rdprimerep, rep):
        for Rprime in range(abs(Rdprime-rm3), abs(Rdprime+rm3)+1, 2):  

            #print ("Reps for C,", Rdprime/2.0, rm3/2.0, Rprime/2.0)


            Mprime = [] 
            Nprime = []
            m6 = []
            n6 = [] 
            Mdprime = []
            Ndprime = [] 

            for x in range (-Rprime, Rprime+1, 2):
                Mprime.append(x/2.0) if x/2.0 not in Mprime else Mprime
                Nprime.append(x/2.0) if x/2.0 not in Nprime else Nprime

            for x in range (-rm3, rm3+1, 2):
                m6.append(x/2.0) if x/2.0 not in m6 else m6
                n6.append(x/2.0) if x/2.0 not in n6 else n6

            for x in range (-Rdprime, Rdprime+1, 2):
                Mdprime.append(x/2.0) if x/2.0 not in Mdprime else Mdprime
                Ndprime.append(x/2.0) if x/2.0 not in Ndprime else Ndprime

            for Mprime_e, Nprime_e, m6_e, n6_e, Mdprime_e, Ndprime_e in itertools.product(Mprime, Nprime, m6, n6, Mdprime, Ndprime):

                i, j, k = index(Rprime,Mprime_e,Nprime_e), index(rm3,m6_e,n6_e), index(Rdprime,Mdprime_e,Ndprime_e) 

                c1 =  CGC((Rdprime/2.0), Mdprime_e, (rm3/2.0), m6_e, (Rprime/2.0), Nprime_e) 
                c2 =  CGC((Rdprime/2.0), Ndprime_e, (rm3/2.0), n6_e, (Rprime/2.0), Mprime_e)
                c3 =  sqrt(Fr((rm3/2.0), beta))/sqrt(Rprime+1.0) 

                C[i][j][k] =  c1 * c2 * c3

    return  C



def makeD(rep, beta):

    D = np.zeros([N_r_p, N_r, N_r])

    for rm1, rm2 in itertools.product(rep, rep):
        for Rdprime in range(abs(rm1-rm2), abs(rm1+rm2)+1, 2):

            #print ("Reps for D,", rm1/2.0, rm2/2.0, Rdprime/2.0)

            m2 = []
            n2 = [] 
            m4 = [] 
            n4 = [] 
            Mdprime = []
            Ndprime = [] 
        
        
            for x in range (-rm1, rm1+1, 2):
                m2.append(x/2.0) if x/2.0 not in m2 else m2
                n2.append(x/2.0) if x/2.0 not in n2 else n2

            for x in range (-rm2, rm2+1, 2):
                m4.append(x/2.0) if x/2.0 not in m4 else m4
                n4.append(x/2.0) if x/2.0 not in n4 else n4

            for x in range (-Rdprime, Rdprime+1, 2):
                Mdprime.append(x/2.0) if x/2.0 not in Mdprime else Mdprime
                Ndprime.append(x/2.0) if x/2.0 not in Ndprime else Ndprime


            for m2_e, n2_e, m4_e, n4_e, Mdprime_e, Ndprime_e in itertools.product(m2, n2, m4, n4, Mdprime, Ndprime):

                i, j, k = index(Rdprime,Mdprime_e,Ndprime_e), index(rm1,m2_e,n2_e), index(rm2,m4_e,n4_e) 
                D[i][j][k] =  CGC((rm1/2.0), m2_e, (rm2/2.0), m4_e, (Rdprime/2.0), Mdprime_e) 
                D[i][j][k] *= CGC((rm1/2.0), n2_e, (rm2/2.0), n4_e, (Rdprime/2.0), Ndprime_e)  
                D[i][j][k] *= sqrt(Fr((rm1/2.0), beta) * Fr((rm2/2.0), beta)) 

    return  D


if __name__ == "__main__":


    beta = np.arange(0.6, 0.70, 0.05).tolist()
    Nsteps = int(np.shape(beta)[0])
    data = np.zeros(Nsteps)

    for p in range (0, Nsteps):


        A = makeA(rep, beta[p])
        B = makeB(rep, beta[p])
        D = makeD(rep, beta[p])
        C = makeC(rep, beta[p])
        '''
        print ("Start norm of A", round(LA.norm(A),10))
        print ("Start norm of B", round(LA.norm(B),10))
        print ("Start norm of C", round(LA.norm(C),10))
        print ("Start norm of D", round(LA.norm(D),10))
        '''
        
        CU = 0.0 

        T = contract('ika,amb,bnc,clj->ijklmn', A, B, C, D)
        norm = np.max(T)
        div = np.sqrt(np.sqrt(norm))

        A  /= div
        B  /= div
        C  /= div
        D  /= div
        CU += np.log(norm)/(2.0)



        for iter in range (Niter):

            A, B, C, D = coarse_graining(A,B,C,D)    
            '''
            print ("A", round(LA.norm(A),10))
            print ("B", round(LA.norm(B),10))
            print ("C", round(LA.norm(C),10))
            print ("D", round(LA.norm(D),10))  
            '''

            T = contract('ika,amb,bnc,clj->ijklmn', A, B, C, D)
            norm = np.max(T)
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
  
                Free = -(1.0/beta[p])*(CU + (np.log(Z)/(2.0**Niter)))
                data[p] = beta[p]*Free 
                print ("f/V =", round(Free,4), "@ beta =", round(beta[p],4), "with D, Niter ->", Dcut, Niter) 
                sys.exit(1)


    if Nsteps > 4:

        dx = beta[1]-beta[0]
        dfdx = np.gradient(data, dx) 
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        f = plt.figure()
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel(r'$\beta$',fontsize=13)
        ax1.set_ylabel('S', color=color,fontsize=13)
        ax1.plot(beta, dfdx, marker="*", color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        plt.grid(True)
        plt.title(r"3d SU(2) PCM model using Triad TRG",fontsize=16, color='black')
        fig.tight_layout()
        plt.show()

    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
    

