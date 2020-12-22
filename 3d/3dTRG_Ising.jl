#!/usr/bin/env julia
# Check for tensor computations:
# https://jutho.github.io/TensorOperations.jl/stable/indexnotation/#Index-notation-with-macros
# A useful MATLAB-Python-Julia cheatsheet is https://cheatsheets.quantecon.org/

import Pkg; Pkg.add("Einsum")
using LinearAlgebra, Statistics, TensorOperations, Einsum, Dates

println("Started: " , Dates.format(now(), "HH:MM:SS "), "on ", today())
Temp = 4.5115;
h = 0.0;
beta = 1.0/Temp;
Niter = 15;
Dcut = 15;

A1 = zeros(2,2,2)
B1 = zeros(2,2,2)
C1 = zeros(2,2,2)
D1 = zeros(2,2,2)

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

    a = sqrt(cosh(beta))
    b = sqrt(sinh(beta))
    W = [a b; a -b]
    #out = np.einsum("ia, ib, ic, id, ie, if -> abcdef", W, W, W, W, W, W)
    #@tensor out[:] := W[1,-1]*W[1,-2]*W[1,-3]*W[1,-4]*W[1,-5]*W[1,-6]
    #Id = Matrix{Float64}(I, 2, 2)
    Id = Matrix(1.0I, 2, 2)
    @einsum A1[x,y,a] = W[a,x] * W[a,y]
    @einsum B1[a,z,b] = Id[a,b] * W[a,z]
    @einsum C1[b,z,c] = Id[b,c] * W[b,z]
    @einsum D1[c,y,x] = W[c,y] * W[c,x]

    return A1, B1, C1, D1

end

function coarse_graining(in1, in2, in3, in4)

    A = in1
    B = in2
    C = in3
    D = in4

    #S2 = contract('dze,izj->diej', B, conjugate(B))
    #@tensoropt S2[d,i,e,j] = B[d,z,e]* conj(B)[i,z,j]
    S2 = @ncon([B, conj(B)],[[-1,1,-3],[-2,1,-4]])

    a = size(S2,1) * size(S2,2)
    b = size(S2,3) * size(S2,4)
    S2 = reshape(S2,(a,b))

    #Tmp = contract('fyx,iyx->fi', D, np.conjugate(D))
    #@tensoropt Tmp[f,i] = D[f,y,x]* conj(D)[i,y,x]
    Tmp = @ncon([D, conj(D)],[[-1,1,2],[-2,1,2]])
    # Seems like we need to replace tensoropt with NCON
    # NCON doesn't need preallocated space. tensoropt does!

    #R2 = contract('ewf,ijk,fk->eiwj', C, np.conjugate(C), Tmp)

    #@tensoropt R2[e,i,w,j] = C[e,w,f]* conj(C)[i,j,k]* Tmp[f,k]
    R2 = @ncon([C, conj(C), Tmp],[[-1,-3,1],[-2,-4,2],[1,2]])
    a = size(R2,1) * size(R2,2)
    b = size(R2,3) * size(R2,4)
    R2mat = reshape(R2,(a,b))

    #S1 = contract('xyd,iyj->xidj', A, np.conjugate(A))
    S1 = @ncon([A, conj(A)],[[-1,1,-3],[-2,1,-4]])
    #@tensoropt S1[x,i,d,j] = A[x,y,d]* conj(A)[i,y,j]
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
    dum = zeros(R2size1,R2size2)
    @einsum dum[x,y] = R2[x,y,b,b]
    R2 = dum

    #R3 = contract('awb,ijk,bk->aiwj', B, np.conjugate(B), Tmp)
    #@tensoropt R3[a,i,w,j] = B[a,w,b]* conj(B)[i,j,k]* Tmp[b,k]
    R3 = @ncon([B, conj(B), R2],[[-1,-3,1],[-2,-4,2],[1,2]])
    a = size(R3,1) * size(R3,2)
    b = size(R3,3) * size(R3,4)
    R3mat = reshape(R3,(a,b))

    #@tensoropt Kprime[i,e] = S1[i,a]*S2[a,b]*R2mat[b,c]*transpose(R3mat)[c,d]*transpose(S1)[d,e]
    Kprime = @ncon([S1,S2,R2mat,transpose(R3mat),transpose(S1)],[[-1,1],[1,2],[2,3],[3,4],[4,-2]])

    a = Int(sqrt(size(Kprime,1)))
    b = Int(sqrt(size(Kprime,2)))
    K = reshape(Kprime,(b,a,b,a))  # K_x1,x2,x3,x4
    U, s1, UL = tensorsvd(K,[1,3],[2,4],Int(Dcut))

    # Now finding "V"
    #S1 = contract('ijk,ipq->jpkq', A, np.conjugate(A))
    #@tensoropt S1[j,p,k,q] = A[i,j,k]* conj(A)[i,p,q]
    S1 = @ncon([A, conj(A)],[[1,-1,-3],[1,-2,-4]])

    a = size(S1,1) * size(S1,2)
    b = size(S1,3) * size(S1,4)
    S1 = reshape(S1,(a,b))

    #R3 = contract('ijk,pqr,kr->ipjq', B, np.conjugate(B), Tmp) # Use 'Tmp' from above

    #@tensoropt R3[i,p,j,q] = B[i,j,k]* conj(B)[p,q,r]* Tmp[k,r]

    R3 = @ncon([B, conj(B), R2],[[-1,-3,1],[-2,-4,2],[1,2]])

    a = size(R3,1) * size(R3,2)
    b = size(R3,3) * size(R3,4)
    R3mat = reshape(R3,(a,b))

    #Kprime = contract('ia,ab,bc,cd,de',S1,S2,R2mat,R3mat.T,S1.T)
    #@tensoropt Kprime[i,e] = S1[i,a]*S2[a,b]*R2mat[b,c]*transpose(R3mat)[c,d]*transpose(S1)[d,e]

    Kprime = @ncon([S1,S2,R2mat,transpose(R3mat),transpose(S1)],[[-1,1],[1,2],[2,3],[3,4],[4,-2]])

    a = Int(sqrt(size(Kprime,1)))
    b = Int(sqrt(size(Kprime,2)))
    K = reshape(Kprime,(b,a,b,a))

    V, s1, VL = tensorsvd(K,[1,3],[2,4],Int(Dcut))
    #@tensoropt Tmp1[c,q,i,x] = D[c,q,p]* U[p,i,x]
    #@tensoropt Tmp2[b,i,q,y] = D[b,j,i]* V[q,j,y]
    #@tensoropt Tmp3[c,x,b,y] = Tmp1[c,q,i,x]* Tmp2[b,i,q,y]

    Tmp1 = @ncon([D, U],[[-1,-2,1],[1,-3,-4]])
    Tmp2 = @ncon([D, V],[[-1,1,-2],[-3,1,-4]])
    Tmp3 = @ncon([Tmp1, Tmp2],[[-1,1,2,-2],[-3,2,1,-4]])

    #Tmp1 = contract('cqp,pix -> cqix',D,U)
    #Tmp2 = contract('bji,qjy -> biqy',D,V)
    #Tmp3 = contract('cqix,biqy -> cxby',Tmp1,Tmp2)
    #Tmp = contract('ijkl,klabc->ijabc', MC, UC)
    #UC = contract('azc,cxby -> abyxz',C,Tmp3)


    MC = @ncon([B, C],[[-1,1,-3],[-2,1,-4]])

    #MC = contract('ijk,pjr->ipkr', B, C)
    #Tmp = contract('ijab,azc,cxby->ijyxz', MC, C, Tmp3)

    Tmp = @ncon([MC, C, Tmp3],[[-1,-2,1,2],[1,-5,3],[3,-4,2,-3]])
    G, st, D = tensorsvd(Tmp,[1,2,3],[4,5],5)

    #G = @ncon([G, st],[[-1,-2,-3,1],[1,-4]])
    @einsum   G[i,j,k,l] = G[i,j,k,a] * st[a,l]
    #G = contract('ijka,al->ijkl', G, st)
    # Was STUCK HERE on 22/12/20 since somehow NCON doesn't like this!~!

    # DC = B_dzb * U*_pix * A_pqa * V*_qjy * A_ijd
    #Tmp1 = contract('pix,pqa->ixqa', np.conjugate(U), A)
    Tmp1 = @ncon([conj(U), A],[[1,-1,-2],[1,-3,-4]])


    Tmp2 = @ncon([conj(V), A],[[-1,1,-2],[-3,1,-4]])
    #Tmp2 = contract('qjy,ijd->qyid', np.conjugate(V), A)

    DC = @ncon([Tmp1, Tmp2],[[1,-1,2,-2],[2,-3,1,-4]])
    #DC = contract('ixqa,qyid->xayd', Tmp1, Tmp2)

    #DC = contract('dzb,xayd->zxyab', B, DC)
    DC = @ncon([B, DC],[[1,-1,-5],[-2,-4,-3,1]])

    Tmp2 = @ncon([DC, G],[[-1,-2,-3,1,2],[1,2,-4,-5]])
    #Tmp2 = contract('ijkab,abmn->ijkmn', DC, G)

    A, st2, MCprime = tensorsvd(Tmp2,[1,2],[3,4,5],Dcut)
    @einsum  MCprime[i,k,l,m] = st2[i,j] * MCprime[j,k,l,m]
    B, st3, C = tensorsvd(MCprime,[1,2],[3,4],Dcut)

    # Split singular piece here!
    sing = st3^0.50 #TODO
    # Can't take square root of matrix. May be linked to NCON error above

    #B = @ncon([B, sing],[[-1,-2,1],[1,-3]])
    #C = @ncon([sing, C],[[-1,1],[1,-2,-3]])
    @einsum  B[i,j,p] = B[i,j,k] * sing[k,p]
    @einsum  C[k,i,p] = sing[k,j] * C[j,i,p]

    return A,B,C,D

end


if Dcut < 4
        error("Increase Dcut")
    end


temp = collect(4.5115:0.02:4.5117)
println(temp)
Nsteps = size(temp,1)
f = zeros(Nsteps)

for p = 1:Nsteps

    A, B, C, D = Z3d_Ising(1.0/temp[p]);
    CU = 0.0;

    for iter = 1:Niter

        A, B, C, D = coarse_graining(A,B,C,D)
        #T = contract('ika,amb,bnc,clj->ijklmn', A, B, C, D)
        T = @ncon([A, B, C, D],[[-1,-3,1],[1,-5,2],[2,-6,3],[3,-4,-2]])
        norm = maximum(T)
        div = sqrt(sqrt(norm))

        A  /= div
        B  /= div
        C  /= div
        D  /= div
        CU += log(norm)/(2.0^(iter))

        if iter == Niter

            Tmp1 = @ncon([A, conj(A)],[[1,2,-1],[1,2,-2]])
            #Tmp1 = contract('dfa,dfj->aj',A,np.conjugate(A))

            Tmp2 = @ncon([D, conj(D)],[[-1,1,2],[-2,1,2]])
            #Tmp2 = contract('cge,mge->cm',D,np.conjugate(D))

            Tmp3 = @ncon([B, conj(B)],[[-1,1,-2],[-3,1,-4]])
            #Tmp3 = contract('ahb,jhk->abjk',B,np.conjugate(B))

            Tmp4 = @ncon([Tmp1, Tmp3],[[1,2],[1,-1,2,-2]])
            #Tmp4 = contract('aj,abjk->bk',Tmp1,Tmp3)

            Tmp5 = @ncon([C, conj(C)],[[-1,1,-2],[-3,1,-4]])
            #Tmp5 = contract('bic,kim->bckm',C,np.conjugate(C))

            #Z = @ncon([Tmp5, Tmp4, Tmp2],[[1,2,3,4],[1,3],[2,4]])
            Z = .0
            @einsum  Z = Tmp5[b,c,k,m] * Tmp4[b,k] * Tmp2[c,m]
            #Z = contract('bckm,bk,cm',Tmp5,Tmp4,Tmp2)
            # Pattern: dfa,ahb,bic,cge,dfj,jhk,kim,mge
            Free = -(temp[p])*(CU + (log(Z)/(2.0^Niter)))
            f[p] = -Free/temp[p]
            println(round(temp[p], digits=8), "  ", round(Free, digits=8))
        end
    end
end

println("Finished: " , Dates.format(now(), "HH:MM:SS "), "on ", today())
