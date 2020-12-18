#!/usr/bin/env julia
# Check for tensor computations:
# https://jutho.github.io/TensorOperations.jl/stable/indexnotation/#Index-notation-with-macros

import Pkg; Pkg.add("Einsum")
using LinearAlgebra, TensorOperations, Einsum, Dates

println("Started: " , Dates.format(now(), "HH:MM:SS "), "on ", today())

Temp = 4.5115;
h = 0.0 
beta = 1.0/Temp;
Niter = 4;
Dcut = 5;

function tensorsvd(input,left,right,chi)
    #Reshape an input tensor into a rectangular matrix with first index corresponding
    #to left set of indices and second index corresponding to right set of indices. Do SVD
    #and then reshape U and V to tensors [left,D] x [D,right]
    #T = np.transpose(input,left+right)

    T = permutedims(input, left+right)


    left_index_list = []
    #for i in range(size(left))
    for i = 1:length(left)
        #left_index_list.append(T.shape[i])
        append!(left_index_list, T.shape[i])
    end


    xsize = prod(left_index_list)
    right_index_list = []

    #for i in range(len(left),len(left)+len(right)):
    for i = length(left):length(left)+length(right)
        #right_index_list.append(T.shape[i])
        append!(right_index_list, T.shape[i])
    end
    ysize = prod(right_index_list)


    T = reshape(T,(xsize,ysize))
    U, s, V = svdfact(T)

    if chi < length(s)
        s = diag(s[:chi])
        U = U[:,:chi]
        V = V[:chi,:]
    else
        chi = length(s)
        s = diag(s)
    end

    U = reshape(U,left_index_list+[chi])
    V = reshape(V,[chi]+right_index_list)

    return U, s, V

end


function Z3d_Ising(beta)

    a = sqrt(cosh(beta))
    b = sqrt(sinh(beta))
    W = [a b; a -b]
    #out = np.einsum("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W)
    #@tensor out[:] := W[1,-1]*W[1,-2]*W[1,-3]*W[1,-4]*W[1,-5]*W[1,-6]
    #Id = Matrix{Float64}(I, 2, 2)
    Id = Matrix(1.0I, 2, 2)
    @einsum A[x,y,a] = W[a,x] * W[a,y]
    @einsum B[a,z,b] = Id[a,b] * W[a,z]
    @einsum C[b,z,c] = Id[b,c] * W[b,z]
    @einsum D[c,y,x] = W[c,y] * W[c,x]

    return A, B, C, D

end

# @tensoropt D[a,b,c,d] := A[a,e,c,f]*B[g,d,e]*C[g,f,b]

function coarse_graining(in1, in2, in3, in4)

    A = in1
    B = in2
    C = in3
    D = in4

    #S2 = contract('dze,izj->diej', B, conjugate(B))

    @tensoropt S2[d,i,e,j] = B[d,z,e]* conj(B)[i,z,j]


    a = size(S2,1) * size(S2,2)
    b = size(S2,3) * size(S2,4)
    S2 = reshape(S2,(a,b))

    #Tmp = contract('fyx,iyx->fi', D, np.conjugate(D))
    @tensoropt Tmp[f,i] = D[f,y,x]* conj(D)[i,y,x]


    #R2 = contract('ewf,ijk,fk->eiwj', C, np.conjugate(C), Tmp)

    @tensoropt R2[e,i,w,j] = C[e,w,f]* conj(C)[i,j,k]* Tmp[f,k]
    a = size(R2,1) * size(R2,2)
    b = size(R2,3) * size(R2,4)
    R2mat = reshape(R2,(a,b))

    #S1 = contract('xyd,iyj->xidj', A, np.conjugate(A))
    @tensoropt S1[x,i,d,j] = A[x,y,d]* conj(A)[i,y,j]

    a = size(S1,1) * size(S1,2)
    b = size(S1,3) * size(S1,4)
    S1 = reshape(S1,(a,b))
    #Tmp = contract('bizz->bi', R2)
    @tensoropt Tmp[b,i] = A[b,i,z,z]

    #R3 = contract('awb,ijk,bk->aiwj', B, np.conjugate(B), Tmp)
    @tensoropt R3[a,i,w,j] = B[a,w,b]* conj(B)[i,j,k]* Tmp[b,k]

    a = size(R3,1) * size(R3,2)
    b = size(R3,3) * size(R3,4)
    R3mat = reshape(R3,(a,b))


    #Kprime = S1 @ S2 @ R2mat @ R3mat.T @ S1.T
    #Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)
    @tensoropt Kprime[i,e] = S1[i,a]*S2[a,b]*R2mat[b,c]*transpose(R3mat)[c,d]*transpose(S1)[d,e]
    # Surprisingly, the above step is prone to some dependence on
    # whethere we use 'matmul', 'dot', '@' and 'contract'

    a = int(sqrt(size(Kprime,1)))
    b = int(sqrt(size(Kprime,2)))
    K = reshape(Kprime,(b,a,b,a))  # K_x1,x2,x3,x4
    U, s1, UL = tensorsvd(K,[0,2],[1,3],int(Dcut))

    # START HERE!!



    return 1

end


if Dcut < 4
        error("Increase Dcut")
    end


dumA = randn(5,5,5,5,5,5)
dumB = randn(5,5,5)
#D = zeros(5,5,5)
#@einsum    D[a,b,c] = A[a,e,f,c,f,g]*B[g,b,e]
dumD = @ncon([dumA, dumB],[[-1,1,2,-3,2,4],[4,-2,1]])
# Prefer 'ncon'. Einsum and NCON gives same answer above~ Checked!

A, B, C, D = Z3d_Ising(beta)
println("Norm of A ", norm(A))
println(size(A,1))
println("Finished: " , Dates.format(now(), "HH:MM:SS "), "on ", today())
