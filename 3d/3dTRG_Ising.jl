#!/usr/bin/env julia
# Check for tensor computations:
# https://jutho.github.io/TensorOperations.jl/stable/indexnotation/#Index-notation-with-macros
# A useful MATLAB-Python-Julia cheatsheet is https://cheatsheets.quantecon.org/

"""
Niter=15 & Dcut=34 takes ~215 seconds while it takes ~262 seconds in Python.
Niter=15 & Dcut=30 takes ~95 seconds while it takes ~118 seconds in Python.
Niter=15 & Dcut=25 takes ~35 seconds while it takes ~43 seconds in Python.
"""

import Pkg;
Pkg.add("Einsum")
Pkg.add("TensorCast")
#Pkg.add("ITensors")
using LinearAlgebra, Statistics, TensorOperations, Einsum, Dates, TensorCast #ITensors

println("Started: " , Dates.format(now(), "HH:MM:SS "), "on ", today())
Temp = 4.5115;
h = 0.0;
beta = 1.0/Temp;
Niter = 15;
Dcut = 30;

macro ltime(expr)
    quote
        print("On Line: ", $(__source__.line), ": ")
        @time $(esc(expr))
    end
end

function tensorsvd(input,left,right,chi)
    #Reshape an input tensor into a rectangular matrix with first index corresponding
    #to left set of indices and second index corresponding to right set of indices. Do SVD
    #and then reshape U and V to tensors [left,D] x [D,right]

    a = Tuple(vcat(left, right)) # Never use append here!
    T = permutedims(input, a)
    left_index_list = []
    for i = 1:length(left)
        append!(left_index_list, size(T, i))
    end

    xsize = prod(left_index_list)
    right_index_list = []

    for i = length(left):length(left)+length(right)-1
        append!(right_index_list, size(T, i+1))
    end
    ysize = prod(right_index_list)

    T = reshape(T,(xsize,ysize))
    F = svd(T)
    U = F.U
    s = Diagonal(F.S)
    V = F.Vt

    if chi < size(s)[1]
        s = s[1:chi,1:chi]
        U = U[:,1:chi]
        V = V[1:chi,:]
    else
        chi = size(s)[1]
        s = s[1:chi,1:chi]
    end

    U = reshape(U, Tuple(vcat(left_index_list, [chi])))
    V = reshape(V, Tuple(vcat([chi],right_index_list)))

    return U, s, V

end

function Z3d_Ising(beta)

    W = [sqrt(cosh(beta)) sqrt(sinh(beta)); sqrt(cosh(beta)) -sqrt(sinh(beta))]
    #out = np.einsum("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W)
    #@tensor out[:] := W[1,-1]*W[1,-2]*W[1,-3]*W[1,-4]*W[1,-5]*W[1,-6]
    #Id = Matrix{Float64}(I, 2, 2)
    Id = Matrix(1.0I, 2, 2)
    @einsum A1[x,y,a] := W[a,x] * W[a,y]
    @einsum B1[a,z,b] := Id[a,b] * W[a,z]
    @einsum C1[b,z,c] := Id[b,c] * W[b,z]
    @einsum D1[c,y,x] := W[c,y] * W[c,x]

    return A1, B1, C1, D1

end

function coarse_graining(in1, in2, in3, in4)

    #S2 = contract('dze,izj->diej', B, conjugate(B))
    @tensoropt S2[d,i,e,j] := in2[d,z,e] * conj(in2)[i,z,j]
    #@ltime S2 = @ncon([in2, conj(in2)],[[-1,1,-3],[-2,1,-4]])

    a = size(S2,1) * size(S2,2)
    b = size(S2,3) * size(S2,4)
    S2 = reshape(S2,(a,b))

    #Tmp = contract('fyx,iyx->fi', D, np.conjugate(D))
    @tensoropt Tmp[f,i] := in4[f,y,x] * conj(in4)[i,y,x]
    #Tmp = @ncon([in4, conj(in4)],[[-1,1,2],[-2,1,2]])
    # Seems like we need to replace tensoropt with NCON
    # NCON doesn't need preallocated space. tensoropt does!

    #R2 = contract('ewf,ijk,fk->eiwj', C, np.conjugate(C), Tmp)

    #@ltime @tensoropt R2[e,i,w,j] := in3[e,w,f]* conj(in3)[i,j,k]* Tmp[f,k]
    R2 = @ncon([in3, conj(in3), Tmp],[[-1,-3,1],[-2,-4,2],[1,2]])
    a = size(R2,1) * size(R2,2)
    b = size(R2,3) * size(R2,4)
    R2mat = reshape(R2,(a,b))

    #S1 = contract('xyd,iyj->xidj', A, np.conjugate(A))
    S1 = @ncon([in1, conj(in1)],[[-1,1,-3],[-2,1,-4]])
    #@ltime @tensoropt S1[x,i,d,j] := in1[x,y,d]* conj(in1)[i,y,j]
    a = size(S1,1) * size(S1,2)
    b = size(S1,3) * size(S1,4)
    S1 = reshape(S1,(a,b))
    #Tmp = contract('bizz->bi', R2)
    #@tensoropt Tmp[b,i] = R2[b,i,z,z]
    #Tmp = @ncon([R2],[[-1,-2,1,1]])
    # This gave an error.
    # do not use `ncon` for less than two tensors

    R2size1 = size(R2)[1]
    R2size2 = size(R2)[2]
    @tensor dum[x,y] := R2[x,y,b,b]

    R2 = dum

    #R3 = contract('awb,ijk,bk->aiwj', B, np.conjugate(B), Tmp)
    #@tensoropt R3[a,i,w,j] = B[a,w,b]* conj(B)[i,j,k]* Tmp[b,k]
    R3 = @ncon([in2, conj(in2), R2],[[-1,-3,1],[-2,-4,2],[1,2]])
    a = size(R3,1) * size(R3,2)
    b = size(R3,3) * size(R3,4)
    R3mat = reshape(R3,(a,b))

    #@tensoropt Kprime[i,e] = S1[i,a]*S2[a,b]*R2mat[b,c]*transpose(R3mat)[c,d]*transpose(S1)[d,e]
    #Kprime = @ncon([S1,S2,R2mat,transpose(R3mat),transpose(S1)],[[-1,1],[1,2],[2,3],[3,4],[4,-2]])
    Kprime = S1 * S2 * R2mat * transpose(R3mat) * transpose(S1)

    a = Int(sqrt(size(Kprime,1)))
    b = Int(sqrt(size(Kprime,2)))
    K = reshape(Kprime,(b,a,b,a))  # K_x1,x2,x3,x4
    U, s1, UL = tensorsvd(K,[1,3],[2,4],Int(Dcut))

    # Now finding "V"
    #S1 = contract('ijk,ipq->jpkq', A, np.conjugate(A))
    #@tensoropt S1[j,p,k,q] = A[i,j,k]* conj(A)[i,p,q]
    S1 = @ncon([in1, conj(in1)],[[1,-1,-3],[1,-2,-4]])

    a = size(S1,1) * size(S1,2)
    b = size(S1,3) * size(S1,4)
    S1 = reshape(S1,(a,b))

    #R3 = contract('ijk,pqr,kr->ipjq', B, np.conjugate(B), Tmp) # Use 'Tmp' from above
    #@tensoropt R3[i,p,j,q] = B[i,j,k]* conj(B)[p,q,r]* Tmp[k,r]

    R3 = @ncon([in2, conj(in2), R2],[[-1,-3,1],[-2,-4,2],[1,2]])

    a = size(R3,1) * size(R3,2)
    b = size(R3,3) * size(R3,4)
    R3mat = reshape(R3,(a,b))

    #Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)
    #@tensoropt Kprime[i,e] = S1[i,a]*S2[a,b]*R2mat[b,c]*transpose(R3mat)[c,d]*transpose(S1)[d,e]
    #Kprime = @ncon([S1,S2,R2mat,transpose(R3mat),transpose(S1)],[[-1,1],[1,2],[2,3],[3,4],[4,-2]])
    Kprime = S1 * S2 * R2mat * transpose(R3mat) * transpose(S1)

    a = Int(sqrt(size(Kprime,1)))
    b = Int(sqrt(size(Kprime,2)))
    K = reshape(Kprime,(b,a,b,a))

    V, s1, VL = tensorsvd(K,[1,3],[2,4],Int(Dcut))

    Tmp1 = @ncon([in4, U],[[-1,-2,1],[1,-3,-4]])
    Tmp2 = @ncon([in4, V],[[-1,1,-2],[-3,1,-4]])
    Tmp3 = @ncon([Tmp1, Tmp2],[[-1,1,2,-2],[-3,2,1,-4]])

    #@tensor Tmp1[c,q,i,x] := in4[c,q,p]* U[p,i,x]
    #@tensor Tmp2[b,i,q,y] := in4[b,j,i]* V[q,j,y]
    #@tensor Tmp3[c,x,b,y] := Tmp1[c,q,i,x]* Tmp2[b,i,q,y]


    MC = @ncon([in2, in3],[[-1,1,-3],[-2,1,-4]])
    Tmp = @ncon([MC, in3, Tmp3],[[-1,-2,1,2],[1,-5,3],[3,-4,2,-3]])
    # *** Expensive!
    #@tensoropt Tmp[i,j,y,x,z] := MC[i,j,a,b]* in3[a,z,c]* Tmp3[c,x,b,y]

    G, st, out4 = tensorsvd(Tmp,[1,2,3],[4,5],Dcut)

    #G = @ncon([G, st],[[-1,-2,-3,1],[1,-4]])
    Gtmp = reshape(G,(size(G,1)*size(G,2)*size(G,3),Int(size(G,4))))
    Gtmp = Gtmp * st
    G = reshape(Gtmp,(size(G,1),size(G,2),size(G,3),Int(size(st,2))))

    #Tmp1 = contract('pix,pqa->ixqa', np.conjugate(U), A)
    Tmp1 = @ncon([conj(U), in1],[[1,-1,-2],[1,-3,-4]])

    Tmp2 = @ncon([conj(V), in1],[[-1,1,-2],[-3,1,-4]])
    #Tmp2 = contract('qjy,ijd->qyid', np.conjugate(V), A)

    DC = @ncon([Tmp1, Tmp2],[[1,-1,2,-2],[2,-3,1,-4]])
    #DC = contract('ixqa,qyid->xayd', Tmp1, Tmp2)

    #DC = contract('dzb,xayd->zxyab', B, DC)
    DC = @ncon([in2, DC],[[1,-1,-5],[-2,-4,-3,1]])

    Tmp2 = @ncon([DC, G],[[-1,-2,-3,1,2],[1,2,-4,-5]])
    #@tensoropt Tmp2[i,j,k,m,n] := DC[i,j,k,a,b]* G[a,b,m,n]
    # *** Expensive!

    out1, st2, MCprime = tensorsvd(Tmp2,[1,2],[3,4,5],Dcut)
    MCprimetmp = reshape(MCprime,(size(MCprime,1),size(MCprime,2)*size(MCprime,3)*size(MCprime,4)))
    MCprimetmp = st2 * MCprimetmp
    MCprime = reshape(MCprimetmp,(size(st2,1),size(MCprime,2),size(MCprime,3),size(MCprime,4)))


    out2, st3, out3 = tensorsvd(MCprime,[1,2],[3,4],Dcut)

    # Split singular piece here!
    sing = st3^0.500 #TODO
    out2tmp = reshape(out2,(size(out2,1)*size(out2,2),size(out2,3)))
    out2tmp = out2tmp * sing
    out2 = reshape(out2tmp,(size(out2,1),size(out2,2),size(sing,2)))

    out3tmp = reshape(out3,(size(out3,1),size(out3,2)*size(out3,3)))
    out3tmp = sing * out3tmp
    out3 = reshape(out3tmp,(size(sing,1),size(out3,2),size(out3,3)))

    return out1,out2,out3,out4

end


if Dcut < 6
        error("Increase Dcut")
    end


temp = collect(4.5115:0.02:4.5117)
Nsteps = size(temp,1)
f = zeros(Nsteps)

for p = 1:Nsteps

    A, B, C, D = Z3d_Ising(1.0/temp[p])
    CU = .0

    for iter = 1:Niter

        A, B, C, D = coarse_graining(A,B,C,D)
        #T = contract('ika,amb,bnc,clj->ijklmn', A, B, C, D)
        #T = @ncon([A, B, C, D],[[-1,-3,1],[1,-5,2],[2,-6,3],[3,-4,-2]])
        #norm = maximum(T)

        norm = maximum(A)*maximum(B)*maximum(C)*maximum(D)
        div = sqrt(sqrt(norm))

        A  /= div
        B  /= div
        C  /= div
        D  /= div
        CU += log(norm)/(2.0^(iter))

        if iter == Niter

            Tmp1 = @ncon([A, conj(A)],[[1,2,-1],[1,2,-2]])
            Tmp2 = @ncon([D, conj(D)],[[-1,1,2],[-2,1,2]])
            Tmp3 = @ncon([B, conj(B)],[[-1,1,-2],[-3,1,-4]])
            Tmp4 = @ncon([Tmp1, Tmp3],[[1,2],[1,-1,2,-2]])
            Tmp5 = @ncon([C, conj(C)],[[-1,1,-2],[-3,1,-4]])

            Z = .0
            @tensor  Z = Tmp5[b,c,k,m] * Tmp4[b,k] * Tmp2[c,m]
            # Pattern: dfa,ahb,bic,cge,dfj,jhk,kim,mge
            Free = -(temp[p])*(CU + (log(Z)/(2.0^Niter)))
            f[p] = -Free/temp[p]
            println(round(temp[p], digits=8), "  ", round(Free, digits=8))
        end
    end
end

println("Finished: " , Dates.format(now(), "HH:MM:SS "), "on ", today())
