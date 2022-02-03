using LinearAlgebra
using SparseArrays

function ham_to_superop(H::AbstractSparseMatrix; convention)
    # See https://arxiv.org/abs/1312.0111, Appendix B.2
    ⊗(A, B) = kron(A, B)
    𝟙 = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, size(H)[1], size(H)[2]))
    H_T = sparse(transpose(H))
    L = sparse(𝟙 ⊗ H - H_T ⊗ 𝟙)
    if convention == :TDSE
        return L
    elseif convention == :LvN
        return 1im * L
    else
        throw(ArgumentError("convention must be :TDSE or :LvN"))
    end
end

function ham_to_superop(H::AbstractMatrix; convention)
    return ham_to_superop(sparse(H); convention=convention)
end


function lindblad_to_superop(A::AbstractSparseMatrix; convention)
    # See https://arxiv.org/abs/1312.0111, Appendix B.2
    ⊗(A, B) = kron(A, B)
    A⁺ = sparse(A')
    A⁺ᵀ = sparse(transpose(A⁺))
    A⁺_A = sparse(A⁺ * A)
    A⁺_A_ᵀ = sparse(transpose(A⁺_A))
    𝟙 = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, size(A)[1], size(A)[2]))
    D = sparse(A⁺ᵀ ⊗ A - (𝟙 ⊗ A⁺_A) / 2 - (A⁺_A_ᵀ ⊗ 𝟙) / 2)
    if convention == :TDSE
        return 1im * D
    elseif convention == :LvN
        return D
    else
        throw(ArgumentError("convention must be :TDSE or :LvN"))
    end
end

function lindblad_to_superop(A::AbstractMatrix; convention)
    return lindblad_to_superop(sparse(A); convention=convention)
end


function dissipator(c_ops; convention)
    N = size(c_ops[1])[1]
    @assert N == size(c_ops[1])[2]
    D = spzeros(ComplexF64, N^2, N^2)
    for A in c_ops
        D += lindblad_to_superop(A; convention=convention)
    end
    return D
end


nhilbert(H::AbstractMatrix) = size(H)[1]
nhilbert(H::Tuple{HT,ET}) where {HT<:AbstractMatrix,ET} = size(H[1])[1]


@doc raw"""Construct a Liouvillian super-operator.

```julia
ℒ = liouvillian(Ĥ, c_ops=(); convention=:LvN)
```

calculates the sparse Liouvillian super-operator `ℒ` from the Hamiltonian `Ĥ`
and a list `c_ops` of Lindblad operators.

With `convention=:LvN`, applying the resulting `ℒ` to a vectorized density
matrix `ρ⃗` calculates ``\frac{d}{dt} \vec{\rho}(t) = ℒ \vec{\rho}(t)``
equivalent to the Liouville-von-Neumann equation for the density matrix ``ρ̂``,

```math
\frac{d}{dt} ρ̂(t)
= -i [Ĥ, ρ̂(t)] + \sum_k\left(
    Â_k ρ̂ Â_k^\dagger
    - \frac{1}{2} A_k^\dagger Â_k ρ̂
    - \frac{1}{2} ρ̂ Â_k^\dagger Â_k
  \right)\,,
```

where the Lindblad operators ``Â_k`` are the elements of `c_ops`.

The Hamiltonian ``Ĥ`` may be time-dependent, using a nested-tuple format by
default, e.g., `(Ĥ₀, (H₁, ϵ₁), (H₂, ϵ₂))`, where `ϵ₁` and `ϵ₂` are functions of
time. In this case, the resulting `ℒ` will also be in nested tuple
format, `ℒ = (ℒ₀, (ℒ₁, ϵ₁), (ℒ₂, ϵ₂))`, where the initial element contains the
superoperator `ℒ₀` for the static component of the Liouvillian, i.e., the
commutator with the drift Hamiltonian `Ĥ₀`, plus the dissipator (sum over
``k``), as a sparse matrix. Time-dependent Lindblad operators are not
supported. The remaining elements are tuples `(ℒ₁, ϵ₁)` and `(ℒ₂, ϵ₂)`
corresponding to the commutators with the two control Hamiltonians, where `ℒ₁`
and `ℒ₂` again are sparse matrices.

If ``Ĥ`` is not time-dependent, the resulting `ℒ` will be a single-element
tuple containing the Liouvillian as a sparse matrix, `ℒ = (ℒ₀, )`.

With `convention=:TDSE`, the Liouvillian will be constructed for the equation
of motion ``-i \hbar \frac{d}{dt} \vec{\rho}(t) = ℒ \vec{\rho}(t)`` to match
exactly the form of the time-dependent Schrödinger equation. While this
notation is not standard in the literature of open quantum systems, it has the
benefit that the resulting `ℒ` can be used in a numerical propagator for a
(non-Hermitian) Schrödinger equation without any change. Thus, for numerical
applications, `convention=:TDSE` is generally preferred. The returned `ℒ`
between the two conventions differs only by a factor of ``i``, since we
generally assume ``\hbar=1``.

The `convention` keyword argument is mandatory, to force a conscious choice.

See [Goerz et. al. "Optimal control theory for a unitary operation under
dissipative evolution", arXiv 1312.0111v2, Appendix
B.2](https://arxiv.org/abs/1312.0111v2) for the explicit construction of the
Liouvillian superoperator as a sparse matrix.
"""
function liouvillian(H::Tuple, c_ops=(); convention)
    L = []
    if length(c_ops) > 0
        L0 = dissipator(c_ops; convention=convention)
    else
        if length(H) > 0
            N = nhilbert(H[1])
            L0 = spzeros(ComplexF64, N^2, N^2)
        else
            throw(ArgumentError("At least one of H and c_ops must be non-empty"))
        end
    end
    for part in H
        if isa(part, Tuple)
            push!(L, (ham_to_superop(part[1], convention=convention), part[2]))
        else
            L0 += ham_to_superop(part, convention=convention)
        end
    end
    return Tuple((L0, L...))
end

function liouvillian(H::AbstractMatrix, c_ops=(); convention)
    return liouvillian((H,), c_ops; convention=convention)
end

function liouvillian(H::Nothing, c_ops=(); convention)
    return liouvillian((), c_ops; convention=convention)
end
