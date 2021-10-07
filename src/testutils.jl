module TestUtils

export random_complex_matrix, random_real_matrix, random_hermitian_matrix
export random_complex_sparse_matrix, random_real_sparse_matrix
export random_hermitian_sparse_matrix, random_state_vector


using Random
using Distributions
using LinearAlgebra
using SparseArrays


"""Construct a random complex matrix of size N×N with spectral radius ρ.

```julia
random_complex_matrix(N, ρ)
```
"""
function random_complex_matrix(N, ρ)
    σ = 1/√N
    d = Normal(0.0, σ)
    H = ρ * (rand(d, (N, N)) + rand(d, (N, N)) * 1im) / √2
end


"""Construct a random real-valued matrix of size N×N with spectral radius ρ.

```julia
random_real_matrix(N, ρ)
```
"""
function random_real_matrix(N, ρ)
    σ = 1/√N
    d = Normal(0.0, σ)
    H = ρ * rand(d, (N, N))
end


"""Construct a random Hermitian matrix of size N×N with spectral radius ρ.

```julia
random_hermitian_matrix(N, ρ)
```
"""
function random_hermitian_matrix(N, ρ)
    σ = 1/√N
    d = Normal(0.0, σ)
    X = rand(d, (N, N))
    H = ρ * (X + X') / (2*√2)
end


"""Construct a random sparse complex matrix.

```julia
random_complex_sparse_matrix(N, ρ, sparsity)
```

returns a matrix of size N×N with spectral radius ρ and the given sparsity
(number between zero and one that is the approximate fraction of non-zero
elements).
"""
function random_complex_sparse_matrix(N, ρ, sparsity)
    σ = 1/√(sparsity * N)
    d = Normal(0.0, σ)
    Hre = sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
    Him = sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
    H = ρ * (Hre + Him * 1im) / √2
end


"""Construct a random sparse real-valued matrix.

```julia
random_real_sparse_matrix(N, ρ, sparsity)
```

returns a matrix of size N×N with spectral radius ρ and the given sparsity
(number between zero and one that is the approximate fraction of non-zero
elements).
"""
function random_real_sparse_matrix(N, ρ, sparsity)
    σ = 1/√(sparsity * N)
    d = Normal(0.0, σ)
    H = ρ * sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
end


"""Construct a random sparse Hermitian matrix.

```julia
random_hermitian_sparse_matrix(N, ρ, sparsity)
```

returns a matrix of size N×N with spectral radius ρ and the given sparsity
(number between zero and one that is the approximate fraction of non-zero
elements).
"""
function random_hermitian_sparse_matrix(N, ρ, sparsity)
    σ = 1/√(sparsity * N)
    d = Normal(0.0, σ)
    H = sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
    return 0.5ρ * (H + H') / √2
end


"""Return a random, normalized Hilbert space state vector of dimension `N`.

```julia
random_state_vector(N)
```
"""
function random_state_vector(N)
    Ψ = rand(N) .* exp.((2π * im) .* rand(N))
    Ψ ./= norm(Ψ)
    return Ψ
end

end
