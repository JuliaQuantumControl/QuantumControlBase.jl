using LinearAlgebra
using SparseArrays

function ham_to_superop(H::AbstractSparseMatrix; convention)
    # See https://arxiv.org/abs/1312.0111, Appendix B.2
    โ(A, B) = kron(A, B)
    ๐ = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, size(H)[1], size(H)[2]))
    H_T = sparse(transpose(H))
    L = sparse(๐ โ H - H_T โ ๐)
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
    โ(A, B) = kron(A, B)
    Aโบ = sparse(A')
    Aโบแต = sparse(transpose(Aโบ))
    Aโบ_A = sparse(Aโบ * A)
    Aโบ_A_แต = sparse(transpose(Aโบ_A))
    ๐ = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, size(A)[1], size(A)[2]))
    D = sparse(Aโบแต โ A - (๐ โ Aโบ_A) / 2 - (Aโบ_A_แต โ ๐) / 2)
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
โ = liouvillian(Hฬ, c_ops=(); convention=:LvN)
```

calculates the sparse Liouvillian super-operator `โ` from the Hamiltonian `Hฬ`
and a list `c_ops` of Lindblad operators.

With `convention=:LvN`, applying the resulting `โ` to a vectorized density
matrix `ฯโ` calculates ``\frac{d}{dt} \vec{\rho}(t) = โ \vec{\rho}(t)``
equivalent to the Liouville-von-Neumann equation for the density matrix ``ฯฬ``,

```math
\frac{d}{dt} ฯฬ(t)
= -i [Hฬ, ฯฬ(t)] + \sum_k\left(
    Aฬ_k ฯฬ Aฬ_k^\dagger
    - \frac{1}{2} A_k^\dagger Aฬ_k ฯฬ
    - \frac{1}{2} ฯฬ Aฬ_k^\dagger Aฬ_k
  \right)\,,
```

where the Lindblad operators ``Aฬ_k`` are the elements of `c_ops`.

The Hamiltonian ``Hฬ`` may be time-dependent, using a nested-tuple format by
default, e.g., `(Hฬโ, (Hโ, ฯตโ), (Hโ, ฯตโ))`, where `ฯตโ` and `ฯตโ` are functions of
time. In this case, the resulting `โ` will also be in nested tuple
format, `โ = (โโ, (โโ, ฯตโ), (โโ, ฯตโ))`, where the initial element contains the
superoperator `โโ` for the static component of the Liouvillian, i.e., the
commutator with the drift Hamiltonian `Hฬโ`, plus the dissipator (sum over
``k``), as a sparse matrix. Time-dependent Lindblad operators are not
supported. The remaining elements are tuples `(โโ, ฯตโ)` and `(โโ, ฯตโ)`
corresponding to the commutators with the two control Hamiltonians, where `โโ`
and `โโ` again are sparse matrices.

If ``Hฬ`` is not time-dependent, the resulting `โ` will be a single-element
tuple containing the Liouvillian as a sparse matrix, `โ = (โโ, )`.

With `convention=:TDSE`, the Liouvillian will be constructed for the equation
of motion ``-i \hbar \frac{d}{dt} \vec{\rho}(t) = โ \vec{\rho}(t)`` to match
exactly the form of the time-dependent Schrรถdinger equation. While this
notation is not standard in the literature of open quantum systems, it has the
benefit that the resulting `โ` can be used in a numerical propagator for a
(non-Hermitian) Schrรถdinger equation without any change. Thus, for numerical
applications, `convention=:TDSE` is generally preferred. The returned `โ`
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
