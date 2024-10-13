# TensorCodes
This repository contains a random collection of codes written mostly in Python & a 3d version of TRG in Julia to study several
observables in spin and gauge models using HOTRG/TNR/Triad algorithms. For index contractions, mostly NCON [introduced in
https://arxiv.org/abs/1402.0939] is used since it is faster than "einsum" or "tensordot" in my tests. Recently, 
I came across "contract" as introduced in https://doi.org/10.21105/joss.00753 which I find is better. 
The different algorithms employed in these codes were introduced in the following papers: 

`HOTRG` in `https://arxiv.org/abs/1201.1144`  

`TNR` in `https://arxiv.org/abs/1412.0732`

`Triad` in `https://arxiv.org/abs/1912.02414` 

As far as I have explored, the bond dimension needed for sufficient convergence in HOTRG is much less than that in the triad formulation. 
The computational cost of triad RG in a three-dimensional model is approximately O(D^6). However, as shown in one of our papers
[arXiv: 2406.10081], ATRG probably does better than triad TRG. So, any future improvement in the algorithm should take that
as an inspiration. It is probably also a reason why I know of no papers written in 4D using triads but several using ATRG
such as [https://arxiv.org/abs/1911.12954]. 

These days, some folks are writing tensor codes in Julia, an example of how index contraction works (in Julia) where I played
around with different options back in 2021 are: 1) @tensor [https://github.com/Jutho/TensorOperations.jl] 2) @einsum [https://github.com/ahwillia/Einsum.jl] 

In particular, in TensorOperations, there is also an added feature for NCON 
[https://jutho.github.io/TensorOperations.jl/stable/indexnotation/#Dynamical-tensor-network-contractions-with-ncon-and-@ncon] 

For example: to execute `A_ijkl` times `B_ijpr` one can use different options as:

```julia
using TensorOperations, Einsum

A = randn(5,5,5,5)
B = randn(5,5,5,5)

@tensor C[k,l,p,r] := A[i,j,k,l] *  B[i,j,p,r]
@einsum C[k,l,p,r] := A[i,j,k,l] *  B[i,j,p,r]
C := @ncon([A, B],[[1,2,-1,-2],[1,2,-3,-4]])
```

To decide which one has better timings and memory allocations, it is useful to time the single-line commands in Julia 
by defining a macro and then calling as below:

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

There is also an additional option of using PRIMME [https://pypi.org/project/primme/] or even SymPy's `svds` as shown below
where `D` is the number of singular values in descending order you want to keep. 

```python 
from scipy.sparse.linalg import svds, eigs
import primme 

U, s, V = svds(T, k=D , which = 'LM')   # Using SciPy
U, s, V = primme.svds(T, D, tol=1e-8, which='LM') # Using PRIMME
s = np.diag(s)

# LM is for keeping large eigenvalues 

# Note that as compared to SciPy's command: U, s, V = sp.linalg.svd(T, full_matrices=False) 
# we have to create a diagonal matrix out of 's' as well if using PRIMME/svds
# It is good idea to provide tolerance to PRIMME's SVD. If you don't then 
# tolerance is 10^4 times the machine precision [approx.~ 10^-12]. 

```

If you get an error like: `LinAlgError: SVD did not converge` then you can try to use: 
`scipy.linalg.svd(..., lapack_driver='gesvd')`

# Cite 

If you used the code `2dXY_HOTRG.py` or `2dXY.py` (or any part of it) or any other code given in `2d` directory, please cite:

```bibtex
@article{Jha:2020oik,
    author = "Jha, Raghav G.",
    title = "{Critical analysis of two-dimensional classical XY model}",
    eprint = "2004.06314",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    doi = "10.1088/1742-5468/aba686",
    journal = "J. Stat. Mech.",
    volume = "2008",
    pages = "083203",
    year = "2020"
}
```

If you used the code `qPotts.ipynb` (or any part of it, say the implementation of triads) given in `3d` directory, please cite:

```bibtex
@article{Jha:2022pgy,
    author = "Jha, Raghav G.",
    title = "{Tensor renormalization of three-dimensional Potts model}",
    eprint = "2201.01789",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    month = "1",
    year = "2022"
}
```


If you used the code `2d_SU2_TRG.py` (or any part of it) or any code given in `2d` directory, please cite:

```bibtex
@article{Bazavov:2019qih,
    author = "Bazavov, Alexei and Catterall, Simon and Jha, Raghav G. and Unmuth-Yockey, Judah",
    title = "{Tensor renormalization group study of the non-Abelian Higgs model in two dimensions}",
    eprint = "1901.11443",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    doi = "10.1103/PhysRevD.99.114507",
    journal = "Phys. Rev. D",
    volume = "99",
    number = "11",
    pages = "114507",
    year = "2019"
}
```

We studied the 3d SU(2) principal chiral model also and the setting up of the initial tensor 
can be found in the `3d_PCM.py` code given in `3d` directory. The code eventually used to produce 
plots in the paper was by Shinichiro and Judah. Please cite our paper if you find it useful. 

```bibtex
@article{Akiyama:2024qgv,
    author = "Akiyama, Shinichiro and Jha, Raghav G. and Unmuth-Yockey, Judah",
    title = "{SU(2) principal chiral model with tensor renormalization group on a cubic lattice}",
    eprint = "2406.10081",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    reportNumber = "JLAB-THY-24-4047, UTCCS-P-154, FERMILAB-PUB-24-0308-T",
    doi = "10.1103/PhysRevD.110.034519",
    journal = "Phys. Rev. D",
    volume = "110",
    number = "3",
    pages = "034519",
    year = "2024"
}
```


If you used the code `3dOO.py` (or any part of it, say the implementation of triads) given in `3d` directory, please cite:

```bibtex
@article{Bloch:2021mjw,
    author = "Bloch, Jacques and Jha, Raghav G. and Lohmayer, Robert and Meister, Maximilian",
    title = "{Tensor renormalization group study of the three-dimensional O(2) model}",
    eprint = "2105.08066",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    doi = "10.1103/PhysRevD.104.094517",
    journal = "Phys. Rev. D",
    volume = "104",
    number = "9",
    pages = "094517",
    year = "2021"
}
```

If you are looking to accelerate some of these codes using GPU, we addressed that in the paper mentioned below
with code at https://zenodo.org/records/8190788

```bibtex
@article{Jha:2023bpn,
    author = "Jha, Raghav G. and Samlodia, Abhishek",
    title = "{GPU-acceleration of tensor renormalization with PyTorch using CUDA}",
    eprint = "2306.00358",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    reportNumber = "JLAB-THY-23-3903",
    doi = "10.1016/j.cpc.2023.108941",
    journal = "Comput. Phys. Commun.",
    volume = "294",
    pages = "108941",
    year = "2024"
}
```

 
Please send questions/suggestions/comments about this repository to raghav.govind.jha@gmail.com
