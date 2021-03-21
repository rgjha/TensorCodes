# This implements triad renormalization network for 3d O(2) model.
# Work in progress. Checks are not complete. 
# Based on algorithm proposed in 1912.02414 
# Most precise critical beta result known from MC: 0.4541647(1)

import sys
import math
from math import sqrt
import numpy as np
import scipy as sp  
from scipy import special
from scipy.special import iv
from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds, eigs
from numpy import linalg as LA
from numpy.linalg import matrix_power
from numpy import ndarray
from matplotlib import pyplot as plt
from itertools import product
import time
import datetime
from opt_einsum import contract


startTime = time.time()
print ("STARTED:" , datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"))  


if len(sys.argv) < 8:
  print("Usage:", str(sys.argv[0]), "<Niter, Dcut, Dn, start, end, incr, h")
  sys.exit(1)

Niter = int(sys.argv[1])
Dcut = int(sys.argv[2])
Dn = int(sys.argv[3])
start = float(sys.argv[4])
end = float(sys.argv[5])
incr = float(sys.argv[6])
h = float(sys.argv[7])
L = np.zeros([2*Dn + 1])
Dcut_triad = int(Dcut*2.) 


def dagger(a):
    return np.conjugate(np.transpose(a))

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


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

    #U, s, V = svds(T, k=D , which = 'LM')   # Using SciPy
    #U, s, V = primme.svds(T, D, which='LM') # Using PRIMME (import primme if using this)
    # LM is for keeping large eigenvalues
    #s = np.diag(s)
    #U, s, V = randomized_svd(T, n_components=D, n_iter=5,random_state=5) # Using scikit-learn 
    # Use "from sklearn.decomposition import TruncatedSVD" 
    # and "from sklearn.utils.extmath import randomized_svd" if using this

    #'''
    U, s, V = sp.linalg.svd(T, full_matrices=False) 
    if D < len(s):
        s = np.diag(s[:D])
        U = U[:,:D]
        V = V[:D,:]
    else:
        D = len(s)
        s = np.diag(s)
    #'''

    U = np.reshape(U,left_index_list+[D])
    V = np.reshape(V,[D]+right_index_list)
        
    return U, s, V


def Z3d(beta, h, Dn):

    if h != 0: 

        for i in range (-Dn,Dn+1):
            L[i+Dn] = np.sqrt(iv(i, beta))
        out = contract("i,j,k,l,m,n->ijklmn", L, L, L, L, L, L)
        out_imp = contract("i,j,k,l,m,n->ijklmn", L, L, L, L, L, L)

        for l,r,u,d,f,b in product(range(-Dn, Dn+1), repeat=6):
            index = l-b-d+u+f-r
            out[l+Dn,b+Dn,d+Dn,u+Dn,f+Dn,r+Dn] *= iv(index, beta*h)
            fac = 0.25*iv(index-2, beta*h) + 0.25*iv(index+2, beta*h) + 0.5*iv(index, beta*h)
            out_imp[l+Dn][b+Dn][d+Dn][u+Dn][f+Dn][r+Dn] *= fac

    
        Tmp1, sing, Tmp2 = tensorsvd(out,[0,1],[2,3,4,5],Dcut_triad) 
        sing = sqrtm(sing)
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

        #T = contract('ija, akb, blc, cmn', A, B, C, D)
        #diff = LA.norm(T) - LA.norm(out)

        #if abs(diff) > 1e-14:  
        #    print ("WARNING: Triads not accurate", diff)
        #    print ("Timestamp: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
        
        Tmp1, sing, Tmp2 = tensorsvd(out_imp,[0,1],[2,3,4,5],Dcut_triad) 
        sing = sqrtm(sing)
        Aimp = contract('ijk,kp->ijp', Tmp1, sing)  
        Tmp2 = contract('ip,pqrst->iqrst', sing, Tmp2)

        Tmp3, stmp2, Tmp4 = tensorsvd(Tmp2,[0,1,2],[3,4],Dcut_triad) 
        sing = sqrtm(stmp2)
        Dimp = contract('kp,pij->kij', sing, Tmp4) 

        Tmp3 = contract('pqrs,sj->pqrj', Tmp3, sing)

        Tmp5, stmp3, Tmp6 = tensorsvd(Tmp3,[0,1],[2,3],Dcut_triad)
        sing = sqrtm(stmp3)
        Bimp = contract('ijk,kp->ijp', Tmp5, sing)
        Cimp = contract('kp,pij->kij', sing, Tmp6)


    else:

        #out = np.zeros(np.repeat(Dcut, 6))
        #for l,r,u,d,f,b in product(range(-Dn, Dn+1), repeat=6):
        #    index = l-b-d+u+f-r
        #    if index == 0:
        #        out[l+Dn,b+Dn,d+Dn,u+Dn,f+Dn,r+Dn] = np.sqrt(iv(l, beta)*iv(r, beta)*iv(u, beta)*iv(d, beta)*iv(f, beta)*iv(b, beta))


        A = np.zeros((Dcut, Dcut, Dcut*2)) 
        B = np.zeros((Dcut*2, Dcut, Dcut*3)) 
        C = np.zeros((Dcut*3, Dcut, Dcut*2))
        D = np.zeros((Dcut*2, Dcut, Dcut))


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
        


        #T = contract('ija, akb, blc, cmn', A, B, C, D) 
        #diff = LA.norm(T) - LA.norm(out)

        #if abs(diff) > 1e-14:  
        #    print ("WARNING: Triads not accurate", diff)
        #    print ("Timestamp: " , datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 

    
    return A, B, C, D, Aimp, Bimp, Cimp, Dimp



def coarse_graining(*args):

    if h == 0:
        [in1,in2,in3,in4] = [args[0],args[1],args[2],args[3]]
    else:
        [in1,in2,in3,in4,in5,in6, in7, in8] = [args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7]]


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
    # DC = B_dzb * U*_pix * A_pqa * V*_qjy * A_ijd
    Tmp1 = contract('pix,pqa->ixqa', np.conjugate(U), in1)
    Tmp2 = contract('qjy,ijd->qyid', np.conjugate(V), in1)
    DC = contract('ixqa,qyid->xayd', Tmp1, Tmp2)
    DC = contract('dzb,xayd->zxyab', in2, DC) 
    Tmp1 = contract('ijkab,abmn->ijkmn', DC, G)
    out1, st2, MCprime = tensorsvd(Tmp1,[0,1],[2,3,4],Dcut) 
    MCprime = contract('ij,jklm->iklm', st2, MCprime)
    out2, st3, out3 = tensorsvd(MCprime,[0,1],[2,3],Dcut)
    # Split singular piece here!
    sing = sqrtm(st3) 
    out2 = contract('ijk,kp->ijp', out2, sing)
    out3 = contract('kj,jip->kip', sing, out3)


    if h != 0: 

        Tmp1 = contract('cqp,pix -> cqix',in8,U)
        Tmp2 = contract('bji,qjy -> biqy',in4,V)
        Tmp3 = contract('cqix,biqy -> cxby',Tmp1,Tmp2)

        MC = contract('ijk,pjr->ipkr', in6, in3)  # MC made!

        Tmp = contract('ijab,azc,cxby->ijyxz', MC, in7, Tmp3) # UC made! 

        G, st, out8 = tensorsvd(Tmp,[0,1,2],[3,4],Dcut) 
        G = contract('ijka,al->ijkl', G, st)  
        # DC = B_dzb * U*_pix * A_pqa * V*_qjy * A_ijd
        Tmp1 = contract('pix,pqa->ixqa', np.conjugate(U), in5)
        Tmp2 = contract('qjy,ijd->qyid', np.conjugate(V), in1)
        DC = contract('ixqa,qyid->xayd', Tmp1, Tmp2)
        DC = contract('dzb,xayd->zxyab', in2, DC)  # DC made! 
        Tmp1 = contract('ijkab,abmn->ijkmn', DC, G)
        out5, st2, MCprime = tensorsvd(Tmp1,[0,1],[2,3,4],Dcut) 
        MCprime = contract('ij,jklm->iklm', st2, MCprime)
        out6, st3, out7 = tensorsvd(MCprime,[0,1],[2,3],Dcut)
        # Split singular piece here!
        sing = sqrtm(st3) 
        out6 = contract('ijk,kp->ijp', out6, sing)
        out7 = contract('kj,jip->kip', sing, out7)

        return out1,out2,out3,out4,out5,out6,out7,out8


    else:
        return out1,out2,out3,out4


if __name__ == "__main__":


    beta = np.arange(start, end, incr).tolist()
    Nsteps = int(np.shape(beta)[0])
    f = np.zeros(Nsteps)
    mag = np.zeros(Nsteps)

    for p in range (0, Nsteps):

        A, B, C, D, Aimp, Bimp, Cimp, Dimp = Z3d(beta[p], h, Dn)

        CU = 0.0 

        for iter in range (Niter):

            if h != 0:
                arr = [A,B,C,D,Aimp,Bimp,Cimp,Dimp]
                A, B, C, D, Aimp, Bimp, Cimp, Dimp = coarse_graining(arr[0],arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7]) 
            else:
                arr = [A,B,C,D]
                A, B, C, D = coarse_graining(arr[0],arr[1],arr[2],arr[3])  

            #T = contract('ika,amb,bnc,clj->ijklmn', A, B, C, D)
            # Expensive step, so bypass! 
            #norm = np.max(T)
            # Alt way to normalize!
            norm = np.max(A)*np.max(B)*np.max(C)*np.max(D) 
            div = np.sqrt(np.sqrt(norm))
            A  /= div
            B  /= div
            C  /= div
            D  /= div
            CU += np.log(norm)/(2.0**(iter+1))

            Aimp /= div
            Bimp /= div
            Cimp /= div 
            Dimp /= div
            

            if iter == Niter-1:

                Tmp1 = contract('dfa,dfj->aj',A,np.conjugate(A))
                Tmp2 = contract('cge,mge->cm',D,np.conjugate(D))
                Tmp3 = contract('ahb,jhk->abjk',B,np.conjugate(B))
                Tmp4 = contract('aj,abjk->bk',Tmp1,Tmp3)
                del Tmp1, Tmp3
                Tmp5 = contract('bic,kim->bckm',C,np.conjugate(C))
                Z = contract('bckm,bk,cm',Tmp5,Tmp4,Tmp2)
                # Pattern: dfa,ahb,bic,cge,dfj,jhk,kim,mge
                # Pattern: T_dfhige * T*_dfhige

                Z_par = CU + (np.log(Z)/(2.0**Niter))
                f[p] = -Z_par
                Free = f[p]*(1.0/beta[p])
                
                # B_ajb * C_bjg * D_gfi -> Z_afi
                # A_iba * Z_afi = T_bf 
                Tmp1 = contract('ajb,bjg->ag',B,C)
                Tmp2 = contract('ag,gfi->afi',Tmp1,D)
                Tmp3 = contract('iba,afi->fb',A,Tmp2)
                X_fp = (np.trace(Tmp3))**2
                X_fp /= np.trace(Tmp3 @ Tmp3)

                if h != 0: 

                    Tmp1 = contract('dfa,dfj->aj',Aimp,np.conjugate(A))
                    Tmp2 = contract('cge,mge->cm',Dimp,np.conjugate(D))
                    Tmp3 = contract('ahb,jhk->abjk',Bimp,np.conjugate(B))
                    Tmp4 = contract('aj,abjk->bk',Tmp1,Tmp3)
                    del Tmp1, Tmp3
                    Tmp5 = contract('bic,kim->bckm',Cimp,np.conjugate(C))
                    Zimp = contract('bckm,bk,cm',Tmp5,Tmp4,Tmp2)
                    mag[p] = Zimp/Z # Or susceptibility (multiply by \beta for latter!) 
                    print (round(beta[p],5),round(f[p],16), round(mag[p],16))
                    file=open("all_3dXY.txt", "a+")
                    file.write("%4.8f \t %2.10e \t  %2.10e \t %2.0f \t %2.0f \t %2.0f \t %2.2e \n" % (beta[p], Free, mag[p], X_fp, Niter, Dcut, Dn, h))
                    file.close()

                else:
                    print (round(beta[p],5),round(f[p],16))

    # Make plots if needed! 
    if Nsteps > 10: 

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
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('T',fontsize=13)
        ax1.set_ylabel('Av. Action', color=color,fontsize=13)
        ax1.plot(beta, data, 'o', color='red');
        ax1.tick_params(axis='y', labelcolor=color)
        plt.title(r"3d Classical model using Triad TRG",fontsize=16, color='black')
        fig.tight_layout()
        outplot = '3dXY' + '_Niter'
        outplot += str(Niter) + '_chi' + str(Dcut)
        outplot += '.pdf'   
        plt.savefig(outplot)

    endTime = time.time() 
    print ("Running time:", round(endTime - startTime, 2),  "seconds")
    print ("COMPLETED:" , datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"))  
    
