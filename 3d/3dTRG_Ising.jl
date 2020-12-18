#!/usr/bin/env julia
# Check for tensor computations:
# https://jutho.github.io/TensorOperations.jl/stable/indexnotation/#Index-notation-with-macros

import Pkg; Pkg.add("Einsum")
using LinearAlgebra, TensorOperations, Einsum, Dates

println("Started: " , Dates.format(now(), "HH:MM:SS "), "on ", today())

Temp = 4.5115;
#h =  float(sys.argv[2])
beta = 1.0/Temp;
Niter = 4;
D = 5;

function tensorsvd(input,left,right,D)
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

    if D < length(s)
        s = diag(s[:D])
        U = U[:,:D]
        V = V[:D,:]
    else
        D = length(s)
        s = diag(s)
    end

    U = reshape(U,left_index_list+[D])
    V = reshape(V,[D]+right_index_list)

    return U, s, V

end


function Z3d_Ising(beta)

    a = sqrt(cosh(beta))
    b = sqrt(sinh(beta))
    W = [a b; a -b]
    #out = np.einsum("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W)
    #@tensor out[:] := W[1,-1]*W[1,-2]*W[1,-3]*W[1,-4]*W[1,-5]*W[1,-6]
    Id = Matrix{Float64}(I, 2, 2)
    @einsum A[x,y,a] = W[a,x] * W[a,y]
    @einsum B[a,z,b] = Id[a,b] * W[a,z]
    @einsum C[b,z,c] = Id[b,c] * W[b,z]
    @einsum D[c,y,x] = W[c,y] * W[c,x]

    return A, B, C, D

end


if D < 4
        error("Increase D")
    end


#alp = randn()
#A = randn(5,5,5,5,5,5)
#B = randn(5,5,5)
#D = zeros(5,5,5)
#@tensor begin
#@einsum    D[a,b,c] = A[a,e,f,c,f,g]*B[g,b,e]
    #D[a,b,c,d] = A[a,e,c,f]*B[g,d,e]*C[g,f,b]
    #E[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e] + alp*C[c,a,b]
#end

A = zeros(2, 2, 2)
B = zeros(2, 2, 2)
C = zeros(2, 2, 2)
D = zeros(2, 2, 2)
A, B, C, D = Z3d_Ising(beta)
println("Norm of A ", norm(A))
println("Finished: " , Dates.format(now(), "HH:MM:SS "), "on ", today())
