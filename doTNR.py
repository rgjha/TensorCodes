#!/usr/bin/python3

import numpy as np
from numpy import linalg as LA
from ncon import ncon


def  doTNR(A,allchi, eigen_tol = 1e-14, maximum_disent_iter = 1000, minimum_iter = 100,  display = True, convergence_tol = 0.01):
    """
------------------------
by Glen Evenbly (c)
------------------------
Implementation of TNR using implicit disentangling. Input 'A' is a four
index tensor that defines the (square-lattice) partition function while
'allchi = [chiM,chiS,chiU,chiH,chiV]' are the bond dimensions.

Optional arguments:
eigen_tol -> threshold for discarding small eigenvalues
maximum_disent_iter -> max iterations in disentangler optimization
minimum_iter -> min iterations in disentangler optimization
display -> display information during optimization
convergence_tol -> halt optimization if changes smaller than convergence_tol
"""

'''
Additional remarks by RGJ
-- This code belongs to the author above. It has been modified and comments have 
been added. doTNR is called by (2d_TNR_method.py). The free energy calculated 
using HOTRG (2d_trg.py) and with TNR method agrees. It is still not unclear
how to embed the Polyakov loop to TNR code. 
'''


    chiHI = A.shape[0]
    chiVI = A.shape[1]
    chiM = min(allchi[0],chiHI*chiVI)
    chiU = min(allchi[2],chiVI)
    #chiH = min(allchi[3],chiHI**2)
    #chiV = min(allchi[4],chiU**2)

    chiH = chiHI**2
    chiV = allchi[4]

    #print ("Shape of A in", np.shape(A))
    

    #-------------------------------------------------------------------------------------------#

    ###### Determine 'q' isometry  ######
    
    qenv = ncon([A,A,A,A,A,A,A,A],[[-1,-2,11,12],[7,8,11,9],[5,12,1,2],[5,9,3,4],
        [-3,-4,13,14],[7,8,13,10],[6,14,1,2],[6,10,3,4]]).reshape(chiHI*chiVI,chiHI*chiVI)
    # qenv_ijkl = A_ijab * A_cdaf * A_gbhm * A_gfrs * A_kltu * A_cdty * A_euhm * A_eyrs 

    dtemp, qtemp = eigCut(qenv, chimax = chiM, eigen_tol = eigen_tol)
    q = qtemp.reshape(chiHI,chiVI,qtemp.shape[1])
    chiM = q.shape[2]
    chiS = min(allchi[1],chiM)

    SP1exact = np.trace(qenv)
    SP1err = abs((SP1exact - np.trace(qtemp.T @ qenv @ qtemp))/SP1exact) 

    # SP1err = | [tr(qenv) - tr(qtemp^dagger * qenv * qtemp)]/tr(qenv) | 

    qA = ncon([q,A],[[1,2,-3],[1,2,-2,-1]])             # qA_qpk = q_ijk * A_ijpq
    C = ncon([qA,qA,qA,qA],[[1,3,-1],[2,3,-2],[1,4,-3],[2,4,-4]])  # C_ijkl = qA_aci * qA_bcj * qA_adk * qA_bdl
    
    
    #-------------------------------------------------------------------------------------------#

    ###### Determine 's' matrix, 'y' isometry, 'u' disentangler  ######
    
    
    u = np.kron(np.eye(chiVI,chiU),np.eye(chiVI,chiU)).reshape(chiVI,chiVI,chiU,chiU)
    # Kronecker or tensor product 
    y = q[:,:u.shape[2],:chiS]
    s= np.eye(q.shape[2],chiS)  # sis a matrix of size 'q.shape[2] x chiS' filled with 1 on a_ii (diagonal)

    Cdub = ncon([C,C],[[-1,-2,1,2],[-3,-4,1,2]])   # Cdub_ijpq = C_ijkl * C_pqkl 
    sCenvD = Cdub.transpose(0,2,1,3)                 # sCenvD =  Cdub_ipjq
    SP2exact = ncon([C,C],[[1,2,3,4],[1,2,3,4]])    # SP2exact  = C_ijkl * C_ijkl
    SP2err = 1

    for k in range(maximum_disent_iter):
        sCenvS = ncon([Cdub,q,q,u,y,y],[[-1,-3,7,8],[1,3,7],[4,6,8],[3,6,2,5],[1,2,-2],[4,5,-4]])
        senvS = ncon([sCenvS,s],[[-1,-2,1,2],[1,2]])
        senvD = ncon([sCenvD,s@ (s.T)],[[-1,-2,1,2],[1,2]])

        if k%100 == 0:
            SP2errnew = abs(1 - (np.trace(senvS @ (s.T))**2) / (np.trace((s.T) @ senvD @ s)*SP2exact))

            # | 1 - {tr[senvS * (s)^dagger]^2 / tr[s^dagger * senvD * s]* SP2exact |
            if k > 50:
                if SP2errnew == 0:
                    SP2errnew = 1e-16 # Hack to stop dividing by zero!
                errdelta = abs(SP2errnew-SP2err) / abs(SP2errnew)
                #print ("SP2errnew is ", SP2errnew)
                if (errdelta < convergence_tol) or (abs(SP2errnew) < 1e-10):
                    SP2err = SP2errnew
                    #if  display:
                    #    print('Iteration: %d of %d, Trunc. Error: %e, %e' % (k,maximum_disent_iter,SP1err,SP2err))
                    break
                
            SP2err = SP2errnew;
            #if  display:
            #    print('Iteration: %d of %d, Trunc. Error: %e, %e' % (k,maximum_disent_iter,SP1err,SP2err))
            
        stemp = LA.pinv(senvD/np.trace(senvD),rcond = eigen_tol) @ senvS
        # LA.pinv computes the (Moore-Penrose) pseudo-inverse of a matrix.
        stemp = stemp/LA.norm(stemp)

        Serrold = abs(1-(np.trace(senvS @ (s.T))**2) / (np.trace((s.T) @ senvD @ s)*SP2exact))


        for p in range(10):
            snew = (1 - 0.1*p)*stemp + 0.1*p*s;
            #print ("SNEW is", snew)
            Serrnew = abs(1 - (ncon([sCenvS,snew,snew],[[1,2,3,4],[1,2],[3,4]])**2)/(ncon([sCenvD,snew @ (snew.T), snew @ (snew.T)],[[1,2,3,4],[1,2],[3,4]])*SP2exact))
            if Serrnew <= Serrold:
                s= snew/LA.norm(snew)
                break
            
        if k > 50:
            yenv = ncon([C,q,q,u,y,s,s,C],[[10,6,3,4],[-1,11,10],[5,8,6],
                         [11,8,-2,9],[5,9,7],[1,-3],[2,7],[1,2,3,4]])
            y = TensorUpdateSVD(yenv,2)

            uenv = ncon([C,q,q,y,y,s,s,C],[[6,9,3,4],[5,-1,6],[8,-2,9],
                         [5,-3,7],[8,-4,10],[1,7],[2,10],[1,2,3,4]])
            uenv = uenv + uenv.transpose(1,0,3,2)
            u = TensorUpdateSVD(uenv,2)         # Disentangler 'u' is ready! Exit loop.

            # Note: Rank(u) = Rank(C) = 4, Rank(q) = Rank(y) = 3, Rank(s) = 2


    Cmod = ncon([C,s,s,s,s],[[1,2,3,4],[1,-1],[2,-2],[3,-3],[4,-4]]) # Cmod_pqrs = C_ijkl * s_ip * s_jq * s_kr * s_ls
    Cnorm = ncon([Cmod,Cmod],[[1,2,3,4],[1,2,3,4]]) / ncon([C,C],[[1,2,3,4],[1,2,3,4]])  # Divide two scalars
    s= s/ (Cnorm**(1/8))


    #-------------------------------------------------------------------------------------------#
    
    ###### Determine 'v' isometry  ######
        
    venv = ncon([y,y,y,y,s,qA,qA,s,s,qA,qA,s,y,y,y,y,s,qA,qA,s,s,qA,qA,s],
                [[1,3,17],[1,4,24],[2,3,18],[2,4,29],[5,17],[7,11,5],[7,12,6],[6,19],
                 [8,18],[10,11,8],[10,12,9],[9,20],[13,15,19],[13,16,25],[14,15,20],[14,16,30],
                 [21,24],[23,-1,21],[23,-2,22],[22,25],[26,29],[28,-3,26],[28,-4,27],[27,30]])


    # tmp1_ijrs = y_ijk * s_pk * (qA)_rsp 
    # tmp2_abcpqr = tmp1_zabc * tmp_zpqr 
    # tmp3_abcdefgh = tmp2_abcdtu * tmp2_efghtu 
    # venv_pqrs = tmp3_abpqcdef * tmp3_abrscdef 


    # Only four indices survive this contraction. They all belong to four qA's 


    venv = 0.5*(venv + venv.transpose(1,0,3,2)).reshape(chiHI**2,chiHI**2)
    dtemp, vtemp = eigCut(venv, chimax = chiH, eigen_tol = eigen_tol)  # Order the eigenvalues and cut
    v = vtemp.reshape(chiHI,chiHI,vtemp.shape[1])

    SP3exact = np.trace(venv)
    SP3err = abs((SP3exact - np.trace(vtemp.T @ venv @ vtemp))/SP3exact)

    # SP3err = | [tr(venv) - tr(vtemp^dagger * venv * vtemp)]/tr(venv) |
    
    
    #-------------------------------------------------------------------------------------------#
    
    ###### Determine 'w' isometry  ######
    
    wenv = ncon([y,y,y,y,s,qA,qA,s,s,qA,qA,s,y,y,y,y,s,qA,qA,s,s,qA,qA,s],
                [[25,-1,26],[25,-2,27],[28,-3,29],[28,-4,30],[1,26],[3,7,1],[3,8,2],[2,13],
                 [4,29],[6,7,4],[6,8,5],[5,14],[9,11,13],[9,12,23],[10,11,14],[10,12,24],
                 [15,27],[17,21,15],[17,22,16],[16,23],[18,30],[20,21,18],[20,22,19],[19,24]]);

    # wenv_pqrs = tmp3_pqabcdef * tmp3_rsabcdef  # See tmp3 above!

    # Only four indices survive this contraction. They all belong to four y's

    wenv = 0.5*(wenv + wenv.transpose(1,0,3,2)).reshape(chiU**2,chiU**2)
    dtemp, wtemp = eigCut(wenv, chimax = chiV, eigen_tol = eigen_tol)
    w = wtemp.reshape(chiU,chiU,wtemp.shape[1])

    SP4exact = np.trace(wenv)
    SP4err = abs((SP4exact - np.trace(wtemp.T @ wenv @ wtemp))/SP4exact)

    # SP4err = | [tr(wenv) - tr(wtemp^dagger * wenv * wtemp)]/tr(wenv) | 

    # The errors for isometries i.e SP1, SP3, SP4 are alike. 

    ###### Generate new 'A' tensor
    Atemp = ncon([v,s,qA,qA,s,w,y,y,v,s,qA,qA,s,w,y,y],
                 [[10,9,-1],[7,19],[6,9,7],[6,10,8],[8,14],[17,18,-2],[16,17,19],[16,18,20],
                  [4,5,-3],[1,20],[3,4,1],[3,5,2],[2,15],[13,12,-4],[11,12,14],[11,13,15]]);

    # Only four indices survive this contraction. They belong each to v, w, v, and w


    
    # wyy_cqr = w_abc * y_paq * y_pbr
    # (qAs)_abd = (qA)_abc * s_cd
    # dum1_czr = v_abc * (qAs)_par * (qAs)_pbz
    # Atemp_adzr = dum1_abc * wyy_dce * dum1_zey * wyy_ryb 
    # Atemp_adzr = v * qA * s * qA * s * w * y * y * v * qA * s * qA * s * w * y * y (without indices!)



    Anorm = LA.norm(Atemp)
    Aout = Atemp / Anorm

    SPerrs = np.array([SP1err,SP2err,SP3err,SP4err])

    #print ("Shape of A out", np.shape(Aout))
    return Aout, q, s,u, y, v, w, Anorm, SPerrs


"""
TensorUpdateSVD: update an isometry or unitary tensor using its (linearized) environment
"""
def TensorUpdateSVD(wIn,leftnum):

    # leftnum is how many elements must it multiply in np.prod (first 2, for ex:)
    # np.prod(wSh) = np.prod(wSh[0:leftnum:1]) * np.prod(wSh[leftnum:len(wSh):1])
    wSh = wIn.shape
    ut,st,vht = LA.svd(wIn.reshape(np.prod(wSh[0:leftnum:1]),np.prod(wSh[leftnum:len(wSh):1])),full_matrices=False)
    return (ut @ vht).reshape(wSh)


"""
eigCut: Eigendecomposition for Hermitian, positive semi-definite matrices.
Keep at most 'chimax' eigenvalues greater than 'eigen_tol'.
"""
def eigCut(rho, chimax = 100000, eigen_tol = 1e-10):
 
    dtemp,utemp = LA.eigh(rho)
    chitemp = min(sum(dtemp>eigen_tol),chimax)
    
    return dtemp[range(-1,-chitemp-1,-1)], utemp[:,range(-1,-chitemp-1,-1)]





    
