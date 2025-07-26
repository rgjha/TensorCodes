# Three-dimensional classical Ising model using ATRG
# ATRG was introduced in the paper: https://arxiv.org/abs/1906.02007

# The work on this code was done by Ranadeep (ranadeep83@gmail.com)
# and Abhishek (asamlodia@gmail.com). If you have questions about this code, it is
# best to ask them (in that order). 

# For now, in 3d, we have three algorithms: HOTRG, Triad, ATRG

# The HOTRG code (not throughly checked) can be found in this repository: 3dTRG.py
# The triad code can be found in this repository: 3dTRG_triad.py 

import os
import sys
import time
import psutil
import numpy as np
import scipy.linalg as LA
from opt_einsum import contract

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        mem_after = process_memory()
        runtime = end-start
        print("function {} took {:.2f} seconds to run and consumed {:.2f} MB of memory".format(func.__name__,
        runtime, float(abs((mem_after - mem_before)))/(1024.0*1024.0)))
        return result
    return wrapper


def initialize_A(temp, h):
    # Make initial tensor of square lattice
    # Ising model at a temperature T
    # normalization of tensor
    beta = 1.0/temp
    c = np.cosh(beta)
    s = np.sinh(beta)
    V = [[(np.exp(beta * h * 0.25) * np.sqrt(c)), (np.exp(beta * h * 0.25) * np.sqrt(s))],
         [(np.exp(-beta * h * 0.25) * np.sqrt(c)), (-np.exp(-beta * h * 0.25) * np.sqrt(s))]]
    T = contract('il, it, iu, ir, ib, id -> lturbd', V, V, V, V, V, V)
    factor = contract('ltultu->',T)
    T /= factor
    del c, s, V
    return T, factor


def svd_first(A, Dcut):
    # SVD of the initial tensor A_lturbd
    # numpy svd returns transpose(V)
    u, s, vt = LA.svd(A.reshape((A.shape[0] * A.shape[1] * A.shape[2]),
                                (A.shape[3] * A.shape[4] * A.shape[5])),
                                full_matrices = False)
    Chi = np.min((s.size, Dcut))  # truncate singular values at D_cut
    U = u[:, :Chi].reshape((A.shape[0], A.shape[1], A.shape[2], Chi))     #U_ltud
    Vt = (vt[:Chi, :]).reshape((Chi, A.shape[3], A.shape[4], A.shape[5])) #Vt_urbd
    S = s[:Chi]
    Vt = np.moveaxis(Vt, 0, -1)                            #Vt_rbdu
    del u, s, vt, Chi
    return U, Vt, S


def svd_main(B, C, Dcut):
    # B_rbdu is (A2_d)rbdu from Update_ATensor
    # C_ltud is (A1_d)ltud from Update_ATensor
    M = contract('rbxu, ltxd -> lturbd', B, C)     # M_lturbd
    M = M.reshape((C.shape[0] * C.shape[1] * B.shape[3]), (B.shape[0] * B.shape[1] * C.shape[3]))
    U, s, VT = LA.svd(M)
    # truncate singular values at D_cut
    Chi = np.min((s.size, Dcut))
    U_t = U[:, 0:Chi].reshape((C.shape[0], C.shape[1], B.shape[3], Chi))         # U_ltud
    VT_t = VT[0:Chi, :].reshape((Chi, B.shape[0], B.shape[1], C.shape[3]))       # Vt_urbd
    s_t = s[0:Chi]
    del M, U, s, VT, Chi
    return U_t, VT_t, s_t


def svd_pq(Tensor, Dcut):
    # svd for P and Q in projector_SVD function
    U, s, VT = LA.svd(Tensor.reshape((Tensor.shape[0] * Tensor.shape[1]),
                                        (Tensor.shape[2] * Tensor.shape[3])))
    # truncate singular values at D_cut
    Chi = np.min((s.size, Dcut))
    U_t = U[:, 0:Chi].reshape((Tensor.shape[0], Tensor.shape[1], Chi))
    V_t = VT[0:Chi, :].reshape((Chi, Tensor.shape[2], Tensor.shape[3]))
    s_t = s[0:Chi]
    return U_t, s_t, V_t


def svd_projector(Y, D, A, X, Dcut, projector):
    # same function as SVD_projector but need to
    # order of bonds via int variable projector
    if projector == 1:
        A = A.transpose(2,1,0,3)                                           # A_utld
        A = A.reshape((A.shape[0] * A.shape[1]), A.shape[2], A.shape[3])   # A_uld

        X = X.transpose(3,1,2,0)                                           # X_dlut
        X = X.reshape((X.shape[0] * X.shape[1]), X.shape[2], X.shape[3])   # X_dut

        Y = Y.transpose(3,1,2,0)                                           # Y_urdb
        Y = Y.reshape((Y.shape[0] * Y.shape[1]), Y.shape[2], Y.shape[3])   # Y_udb

        D = D.transpose(2,1,0,3)                                           # D_drbu
        D = D.reshape((D.shape[0] * D.shape[1]), D.shape[2], D.shape[3])   # D_dbu

    elif projector == 2:
        A = A.transpose(2,0,1,3)                                           # A_utld
        A = A.reshape((A.shape[0] * A.shape[1]), A.shape[2], A.shape[3])   # A_uld

        X = X.transpose(3,0,2,1)                                           # X_dlut
        X = X.reshape((X.shape[0] * X.shape[1]), X.shape[2], X.shape[3])   # X_dut

        Y = Y.transpose(3,0,2,1)                                           # Y_urdb
        Y = Y.reshape((Y.shape[0] * Y.shape[1]), Y.shape[2], Y.shape[3])   # Y_udb

        D = D.transpose(2,0,1,3)                                           # D_drbu
        D = D.reshape((D.shape[0] * D.shape[1]), D.shape[2], D.shape[3])   # D_dbu

    else:
        print("Error! Wrong projector number specified")
        sys.exit(1)

    P = contract('xli, xrj, yiL, yjR -> lLrR', A, A, X, X)
    Q = contract('xil, xjr, yLi, yRj -> lLrR', Y, Y, D, D)
    Up_t, sp_t, Vp_t = svd_pq(P, Dcut)
    Uq_t, sq_t, Vq_t = svd_pq(Q, Dcut)

    N = contract('ijk, ijl -> kl', Up_t, Uq_t)
    Un, sn, VnT = LA.svd(N)
    sn_new = np.diag(np.sqrt(sn))

    E = contract('ij, jk, ilm -> lmk', Un, sn_new, Vp_t)
    F = contract('ij, jk, ilm -> lmk', np.transpose(VnT), sn_new, Vq_t)

    del A, X, Y, D, P, Q, Up_t, sp_t, Vp_t, Uq_t, sq_t, Vq_t, N, Un, sn, VnT, sn_new
    return E, F


def make_new_GH(Tensor):
    # performs some of the repeated
    # operations of make_next_A
    Tensor_new = Tensor.reshape((Tensor.shape[0] * Tensor.shape[1]), Tensor.shape[2], Tensor.shape[3])
    UT, sT, VtT = LA.svd(Tensor_new.reshape((Tensor_new.shape[0] * Tensor_new.shape[1]), Tensor_new.shape[2]),
                         full_matrices = False)
    UT = UT.reshape(Tensor_new.shape[0], Tensor_new.shape[1], Tensor_new.shape[2])
    return UT, sT, VtT


def make_next_A(G,H):
    # G_ltud
    # H_rbdu
    Ug, sg, VTg = make_new_GH(G)
    Uh, sh, VTh = make_new_GH(H)
    #K = contract('ij, jk, kl, lm -> im', np.diag(sg), VTg, np.transpose(VTh), np.diag(sh))   # K_ud
    K = np.diag(sg) @ VTg @ np.transpose(VTh) @ np.diag(sh)
    Uk, sk, VTk = LA.svd(K)
    A1 = contract('Lux, xd -> Lud', Ug, Uk)
    A2 = contract('ux, Rdx -> Rdu', VTk, Uh)
    A1 = A1.reshape(G.shape[0], G.shape[1], A1.shape[1], A1.shape[2])  # A1_ltud
    A2 = A2.reshape(H.shape[0], H.shape[1], A2.shape[1], A2.shape[2])  # A_2rbdu
    A1 = A1.transpose(1, 2, 0, 3)  # A1_tuld
    A2 = A2.transpose(1, 2, 0, 3)  # A2_bdru
    sd = sk
    del Ug, sg, VTg, Uh, sh, VTh, K, Uk, VTk
    import gc
    gc.collect()
    return A1, A2, sd

def coarse_graining(A, D, sd, Dcut):
    # After first svd : A = U_ltud, D = Vt_rbdu
    # A_ltud
    # B_rbdu
    # C_ltud
    # D_rbdu
    C = contract('ltux, xd -> ltud', A, np.diag(sd))   #  C_ltud
    B = contract('ux, rbdx -> rbdu', np.diag(sd), D)   #  B_rbdu
    U, VT, s = svd_main(B, C, 2*Dcut)                  #  Form M matrix and do its SVD
    # U_ltux
    # s_xy
    # VT_yrbd
    Y = contract('ux, xrbd -> rbdu', np.diag(np.sqrt(s)), VT)         # Y_rbdu
    X = contract('ltux, xd -> ltud', U, np.diag(np.sqrt(s)))          # X_ltud
    E1, F1 = svd_projector(Y,D,A,X,Dcut, 1)
    E2, F2 = svd_projector(Y,D,A,X,Dcut, 2)
    G = contract('ijux, IJxd, iIl, jJt ->ltud', A, X, E1, E2, optimize='optimal')
    H = contract('ijxu, IJdx, iIr, jJb ->rbdu', Y, D, F1, F2, optimize='optimal')
    A1_new, A2_new, sd_new = make_next_A(G, H)
    factor = contract('tulx, xy, tuly -> ', A1_new, np.diag(sd_new), A2_new)
    sd_new /= factor
    del C, B, U, VT, s, Y, X, E1, E2, G, H
    return A1_new, A2_new, sd_new, factor

def main(T,h,D,TRG_steps):
    # Initialization
    C = 0.0
    N = 1
    A, factor = initialize_A(T, h)

    # TRG_factors = [factor]
    C = np.log(factor)

    # TRG iteration
    for i_TRG in range(TRG_steps):
        print("Iteration : ", i_TRG+1)
        if i_TRG == 0:
            A1, A2, sd = svd_first(A, D)

        A1, A2, sd, factor = coarse_graining(A1, A2, sd, D)

        C = np.log(factor) + 2.0*C
        N *= 2.0

        if(i_TRG == TRG_steps-1):
            A1, A2, sd, factor = coarse_graining(A1, A2, sd, D)
            Ttensor = contract('tulx, xy, bdry -> tulbdr', A1, np.diag(sd), A2)
            num =  contract('ii->',Ttensor.reshape((Ttensor.shape[0]*Ttensor.shape[1]*Ttensor.shape[2], Ttensor.shape[3]*Ttensor.shape[4]*Ttensor.shape[5])))
            den  = contract('ij,jk->ik',Ttensor.reshape((Ttensor.shape[0]*Ttensor.shape[1]*Ttensor.shape[2], Ttensor.shape[3]*Ttensor.shape[4]*Ttensor.shape[5])),
            Ttensor.reshape((Ttensor.shape[0]*Ttensor.shape[1]*Ttensor.shape[2], Ttensor.shape[3]*Ttensor.shape[4]*Ttensor.shape[5])))
            X = ((num*num)/contract('ii->',den))
            C = np.log(factor) + 2*C
            N *= 2.0
            free_energy_density = -T*((np.log(factor)) + C) / (N)

    return free_energy_density, X

Dcut = 15
iterations = 12
Temp = 4.5115 # Tc for 3d Ising on cubic lattice, f ~ -3.51
h = 0.0
freeeden, X = main(Temp, h, Dcut, iterations)
print("\nDcut : {}, h : {}, Temp : {}".format(Dcut, h, Temp))
print("Total iterations : ", iterations)
print("Free Energy Density : ", freeeden)
print("X value : ", X)

