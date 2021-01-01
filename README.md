# TensorCodes
This repository contains a random collection of codes written in Python & Julia (some are work in progress) to study several
observables in models using HOTRG/TNR/Triad algorithms. For index contractions, mostly NCON [introduced in
https://arxiv.org/abs/1402.0939] is used
since it is faster than "einsum" or "tensordot" in my tests. Recently, I came across "contract"
as introduced first in https://doi.org/10.21105/joss.00753. The different algorithms employed in these codes 
were introduced in the following papers: 

HOTRG --> https://arxiv.org/abs/1201.1144  

TNR --> https://arxiv.org/abs/1412.0732

Triad --> https://arxiv.org/abs/1912.02414

Example of how index contraction works (in Julia) where I have been playing around with different options: 1) @tensor [https://github.com/Jutho/TensorOperations.jl] 2) @einsum [https://github.com/ahwillia/Einsum.jl] 

In particular, in TensorOperations, there is also added feature for NCON [https://jutho.github.io/TensorOperations.jl/stable/indexnotation/#Dynamical-tensor-network-contractions-with-ncon-and-@ncon] 

For example: to execute `A_ijkl` times `B_ijpr` one can use different options as:

```julia
using TensorOperations, Einsum

A = randn(5,5,5,5)
B = randn(5,5,5,5)

@tensor C[k,l,p,r] := A[i,j,k,l] *  B[i,j,p,r]
@einsum C[k,l,p,r] := A[i,j,k,l] *  B[i,j,p,r]
C := @ncon([A, B],[[1,2,-1,-2],[1,2,-3,-4]])
```

To decide which one has better timings and memory allocations, it is useful to time the single-line commands in Julia by defining a macro 
and then calling as below:

```julia
using TensorOperations, Einsum

macro ltime(expr)
    quote
        print("On Line: ", $(__source__.line), ": ")
        @time $(esc(expr))
    end
end

@ltime @tensor C[k,l,p,r] := A[i,j,k,l] *  B[i,j,p,r]
```


Similarly in Python, we will have

```python 
from opt_einsum import contract
from ncon import ncon 
import numpy as np 

A = np.random.rand(5,5,5,5)
B = np.random.rand(5,5,5,5)

C = contract('ijkl,ijpr->klpr', A, B)
C = np.einsum('ijkl,ijpr->klpr', A, B)
C = ncon([A, B],[[1,2,-1,-2],[1,2,-3,-4]])

# More complicated contraction example is one given below:

input = np.random.rand(5,5,5,5,5,5)
Ux = Uy = np.random.rand(5,5,5)

out = ncon([Ux,Uy,input,input,Uy,Ux],[[3,4,-2],[1,2,-1],[1,3,5,7,10,-6],[2,4,6,8,-5,10],[5,6,-3],[7,8,-4]])
```

The single most crucial and expensive step in these tensor computations is the `SVD` (singular value decomposition). 
For a matrix of size `m` x `n` with `m > n`, the cost scales like O(mn^2). Depending on how many
singular values (arranged in descending order) we keep, the cost can vary. In 2009, a new 
method called 'randomized SVD' was proposed by Halko et al. [https://arxiv.org/abs/0909.4061]. 
This has already been useful for machine learning purposes and the implementation has been done 
in scikit-learn [https://scikit-learn.org/stable/]. Use this with caution! 

```python 
from sklearn.utils.extmath import randomized_svd
U, s, V = randomized_svd(T, n_components=D, n_iter=5,random_state=5)
```


Please send questions/suggestions about this repository to rgjha1989@gmail.com
