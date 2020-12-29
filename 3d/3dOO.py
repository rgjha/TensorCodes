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
  print("Usage:", str(sys.argv[0]), "<Niter, Dcut, min_cut")
  sys.exit(1)

Niter = int(sys.argv[1])
Dcut = int(sys.argv[2])
min_cut = float(sys.argv[3])
Dcut_triad = int(Dcut*1.5) 

if Dcut%2 == 0:
    print ("Dcut must be odd for now")
    sys.exit(1) 


L = np.zeros([Dcut])
Dn = int(Dcut/2.0)


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


def Z3d(beta, h, Dn):

    betah = beta*h 

    for i in range (-Dn,Dn+1):
        L[i+Dn] = np.sqrt(sp.special.iv(i, beta))
    out = contract("i,j,k,l,m,n->ijklmn", L, L, L, L, L, L)


    if min_cut != 0.0:

        p31 = np.asarray(out)
        print ("Min. element of T", np.min(out))
        p31[p31 < min_cut] = 0.0
        index_l = np.nonzero(p31)[0]
        index_r = np.nonzero(p31)[1]
        index_u = np.nonzero(p31)[2]
        index_d = np.nonzero(p31)[3]
        index_f = np.nonzero(p31)[4]
        index_b = np.nonzero(p31)[5]

        length = len(index_l)
        frac = ((Dcut**6)-length)*100/(Dcut**6)
        if frac == 0.00:
            print ("No truncation apart from Dcut")
        else:
            print ("Truncating " "%8.7f" "%% of the initial tensor" %(frac))

        for iter in range (0, length):

            l = index_l[iter]
            r = index_r[iter]
            u = index_u[iter]
            d = index_d[iter]
            f = index_f[iter]
            b = index_b[iter]
            index = l-r+u-d+f-b
            out[l][r][u][d][f][b] *= sp.special.iv(index, betah)


    else: 

        for l in range (-Dn,Dn+1):
            for r in range (-Dn,Dn+1):
                for u in range (-Dn,Dn+1):
                    for d in range (-Dn,Dn+1):
                        for f in range (-Dn,Dn+1):
                            for b in range (-Dn,Dn+1):

                                index = l-r+u-d+f-b
                                out[l+Dn][r+Dn][u+Dn][d+Dn][f+Dn][b+Dn] *= sp.special.iv(index, betah)


    out = out.transpose(0,5,3,2,4,1)  
    # lrudfb to lbdufr
    #f.write ("AM ", psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

    Tmp1, stmp1, Tmp2 = tensorsvd(out,[0,1],[2,3,4,5],Dcut_triad) 
    sing = sqrtm(stmp1)
    A = contract('ijk,kp->ijp', Tmp1, sing)  
    Tmp2 = contract('ip,pqrst->iqrst', sing, Tmp2)

    Tmp3, stmp2, Tmp4 = tensorsvd(Tmp2,[0,1,2],[3,4],Dcut_triad) 
    sing = sqrtm(stmp2)
    D = contract('kp,pij->kij', sing, Tmp4) 

    Tmp3 = contract('pqrs,sj->pqrj', Tmp3, sing)

    Tmp5, stmp3, Tmp6 = tensorsvd(Tmp3,[0,1],[2,3],Dcut_triad)
    sing = sqrtm(stmp3)
    B = contract('ijk,kp->ijp', Tmp5, sing)
    C = contract('kp,pij->kij', sing, Tmp6)

    T = contract('ika,amb,bnc,clj->ijklmn', A, B, C, D)
    diff = LA.norm(T) - LA.norm(out)

    if abs(diff) > 1e-14:  
        print ("Error: Triads not accurate", diff)
        print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 

    return A, B, C, D


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
    del Tmp
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

    dum1 = S2 @ R2mat 
    Kprime = S1 @ dum1 @ R3mat.T @ S1.T
    #Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)

    a = int(np.sqrt(np.shape(Kprime)[0]))
    b = int(np.sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,a,b,a))         
    U, s1, UL = tensorsvd(K,[0,2],[1,3],int(Dcut)) 


    S1 = contract('ijk,ipq->jpkq', A, np.conjugate(A))
    a = np.shape(S1)[0] * np.shape(S1)[1]
    b = np.shape(S1)[2] * np.shape(S1)[3]
    S1 = np.reshape(S1,(a,b))
    R3 = contract('ijk,pqr,kr->ipjq', B, np.conjugate(B), Tmp) # Use 'Tmp' from above
    a = np.shape(R3)[0] * np.shape(R3)[1]
    b = np.shape(R3)[2] * np.shape(R3)[3]
    R3mat = np.reshape(R3,(a,b))


    #Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)
    Kprime = S1 @ dum1 @ R3mat.T @ S1.T # Use Tmp from above
    del dum1 

    a = int(np.sqrt(np.shape(Kprime)[0]))
    b = int(np.sqrt(np.shape(Kprime)[1]))
    K = np.reshape(Kprime,(b,a,b,a))
    V, s1, VL = tensorsvd(K,[0,2],[1,3],Dcut)

    # Free some arrays which are no longer needed  
    del K
    del Kprime
    del S1 
    del R3 

    Tmp1 = contract('cqp,pix -> cqix',D,U)
    Tmp2 = contract('bji,qjy -> biqy',D,V)
    Tmp3 = contract('cqix,biqy -> cxby',Tmp1,Tmp2)
    del Tmp1, Tmp2
    
    MC = contract('ijk,pjr->ipkr', B, C)
    Tmp = contract('ijab,azc,cxby->ijyxz', MC, C, Tmp3)

    G, st, D = tensorsvd(Tmp,[0,1,2],[3,4],Dcut) 
    G = contract('ijka,al->ijkl', G, st)  

    del Tmp, Tmp3

    # DC = B_dzb * U*_pix * A_pqa * V*_qjy * A_ijd
    Tmp1 = contract('pix,pqa->ixqa', np.conjugate(U), A)
    Tmp2 = contract('qjy,ijd->qyid', np.conjugate(V), A)
    DC = contract('ixqa,qyid->xayd', Tmp1, Tmp2)
    DC = contract('dzb,xayd->zxyab', B, DC)

    del Tmp2
    del Tmp1
 
    Tmp1 = contract('ijkab,abmn->ijkmn', DC, G)
    A, st2, MCprime = tensorsvd(Tmp1,[0,1],[2,3,4],Dcut) 

    del Tmp1

    MCprime = contract('ij,jklm->iklm', st2, MCprime)
    B, st3, C = tensorsvd(MCprime,[0,1],[2,3],Dcut)

    del MCprime


    # Split singular piece here!
    sing = sqrtm(st3) 
    B = contract('ijk,kp->ijp', B, sing)
    C = contract('kj,jip->kip', sing, C)

    return A,B,C,D 



if __name__ == "__main__":

    #f=open("output_3d_XY_testing.txt", "a+")

    beta = np.arange(0.55, 0.6, 0.1).tolist()
    Nsteps = int(np.shape(beta)[0])
    f = np.zeros(Nsteps)

    for p in range (0, Nsteps):

        A, B, C, D  = Z3d(beta[p], 0.0, Dn)
        CU = 0.0 

        for iter in range (Niter):

            A, B, C, D = coarse_graining(A,B,C,D)  
            #f.write("One CG step")
            
            #T = contract('ika,amb,bnc,clj->ijklmn', A, B, C, D)
            #norm = np.max(T)

            # Alt way to normalize!
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
                del Tmp1, Tmp3
                Tmp5 = contract('bic,kim->bckm',C,np.conjugate(C))
                Z = contract('bckm,bk,cm',Tmp5,Tmp4,Tmp2)
                # Pattern: dfa,ahb,bic,cge,dfj,jhk,kim,mge

                Z_par = CU + (np.log(Z)/(2.0**Niter))
                f[p] = -Z_par
                Free = f[p]*(1.0/beta[p])
                print (round(beta[p],5),round(f[p],16),round(Free,16))
                #f.write(round(beta[p],5),round(f[p],16))


    #f.close()

    # Make plots if needed! 
    if Nsteps > 3: 

        dx = beta[1]-beta[0] # Assuming equal spacing ...
        dfdx = np.gradient(f, dx) 
        d2fdx2 = np.gradient(dfdx, dx) 
        out = [] 
        for i in range(0, len(dfdx)): 
            out.append(dfdx[i]*(-1/3.0)) 
        out1 = [] 
        for i in range(0, len(d2fdx2)):
            out1.append(d2fdx2[i])

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        data = out   

        f = plt.figure()
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('T',fontsize=13)
        ax1.set_ylabel('Av. Action', color=color,fontsize=13)
        #ax1.plot(beta, data, '.r-')
        ax1.plot(beta, data, 'o', color='red');
        ax1.tick_params(axis='y', labelcolor=color)
        plt.title(r"3d Classical model using Triad TRG",fontsize=16, color='black')
        fig.tight_layout()

        outplot = '3dXY' + '_Niter'
        outplot += str(Niter) + '_chi' + str(Dcut)
        outplot += '.pdf'   

        plt.savefig(outplot)


    print ("COMPLETED: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
    
