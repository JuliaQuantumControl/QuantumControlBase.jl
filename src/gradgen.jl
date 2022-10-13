import LinearAlgebra
import QuantumPropagators
import Base: -, *

using QuantumPropagators.Generators:
    getcontrols, getcontrolderivs, evalcontrols, evalcontrols!


@doc raw"""Extended generator for the standard dynamic gradient.

```julia
G̃ = GradGenerator(G)
```

contains the original time-dependent generator `G` (a Hamiltonian or
Liouvillian) in `G̃.G`, a vector of control derivatives ``∂G/∂ϵₗ(t)`` in
`G̃.control_derivs`, and the controls in `G̃.controls`.

For a generator ``G = Ĥ(t) = Ĥ₀ + ϵ₁(t) Ĥ₁ + … +  ϵₙ(t) Ĥₙ``, this extended
generator encodes the block-matrix

```math
G̃ = \begin{pmatrix}
         Ĥ(t)  &  0    &  \dots   &  0     &  Ĥ₁     \\
         0     &  Ĥ(t) &  \dots   &  0     &  Ĥ₂     \\
    \vdots     &       &  \ddots  &        &  \vdots \\
         0     &  0    &  \dots   &  Ĥ(t)  &  Ĥₙ     \\
         0     &  0    &  \dots   &  0     &  Ĥ(t)
\end{pmatrix}
```

Note that the ``∂G/∂ϵₗ(t)`` (``Ĥₗ`` in the above example) may be
time-dependent, to account for the possibility of non-linear control terms, see
[`getcontrolderiv`](@ref QuantumPropagators.Generators.getcontrolderiv).
"""
struct GradGenerator{GT,CDT,CT}
    G::GT
    control_derivs::Vector{CDT}
    controls::Vector{CT}

    function GradGenerator(G::GT) where {GT}
        controls = collect(getcontrols(G))
        control_derivs = getcontrolderivs(G, controls)
        CT = eltype(controls)
        CDT = eltype(control_derivs)
        new{GT,CDT,CT}(G, control_derivs, controls)
    end

end


"""Static generator for the dynamic gradient.

```julia
G::GradgenOperator = evalcontrols(gradgen::GradGenerator, vals_dict)
```

is the result of plugging in specific values for all controls in a
[`GradGenerator`](@ref). See [`evalcontrols`](@ref
QuantumPropagators.Generators.evalcontrols) and [`evalcontrols!`](@ref
QuantumPropagators.Generators.evalcontrols!).

The resulting object can be multiplied directly with a [`GradVector`](@ref),
e.g., in the process of evaluating a piecewise-constant time propagation.
"""
struct GradgenOperator{num_controls,GT,CGT}
    G::GT
    control_deriv_ops::Vector{CGT}
end

# Dummy initializer: this creates a GradgenOperator that fits a
# GradGenerator structurally
function GradgenOperator(gradgen::GradGenerator)
    dummy_vals = IdDict(control => 1.0 for control in gradgen.controls)
    dummy_tlist = [0.0, 1.0]
    G = evalcontrols(gradgen.G, dummy_vals, dummy_tlist, 1)
    control_deriv_ops =
        [evalcontrols(μ, dummy_vals, dummy_tlist, 1) for μ in gradgen.control_derivs]
    num_controls = length(control_deriv_ops)
    GT = typeof(G)
    CGT = eltype(control_deriv_ops)
    GradgenOperator{num_controls,GT,CGT}(G, control_deriv_ops)
end


function QuantumPropagators.Generators.getcontrols(gradgen::GradGenerator)
    return getcontrols(gradgen.G)
end

QuantumPropagators.Generators.evalcontrols(O::GradgenOperator, _...) = O
QuantumPropagators.Generators.evalcontrols!(O1::T, O2::T, _...) where {T<:GradgenOperator} =
    O1
QuantumPropagators.Generators.getcontrols(O1::GradgenOperator) = Tuple([])


function QuantumPropagators.Generators.evalcontrols!(
    G::GradgenOperator,
    gradgen::GradGenerator,
    vals_dict::AbstractDict,
    args...
)
    evalcontrols!(G.G, gradgen.G, vals_dict, args...)
    for (i, control) in enumerate(gradgen.controls)
        μ = gradgen.control_derivs[i]
        G.control_deriv_ops[i] = evalcontrols(μ, vals_dict, args...)
        # In most cases (for linear controls), evalcontrols(μ, ...) = μ
        # Hence, we're not using `copyto!`.
    end
    return G
end


function QuantumPropagators.Generators.evalcontrols(
    gradgen::GradGenerator,
    vals_dict::AbstractDict,
    args...
)
    G = GradgenOperator(gradgen)
    QuantumPropagators.Generators.evalcontrols!(G, gradgen, vals_dict, args...)
end


@doc raw"""Extended state-vector for the dynamic gradient.

```julia
Ψ̃ = GradVector(Ψ, num_controls)
```

for an initial state `Ψ` and `num_controls` control fields.

The `GradVector` conceptually corresponds to a direct-sum (block) column-vector
``Ψ̃ = (|Ψ̃₁⟩, |Ψ̃₂⟩, … |Ψ̃ₙ⟩, |Ψ⟩)^T``, where ``n`` is `num_controls`. With a
matching ``G̃`` as in the documentation of [`GradGenerator`](@ref),
we have

```math
G̃ Ψ̃ = \begin{pmatrix}
Ĥ |Ψ̃₁⟩ + Ĥ₁|Ψ⟩ \\
\vdots \\
Ĥ |Ψ̃ₙ⟩ + Ĥₙ|Ψ⟩ \\
Ĥ |Ψ⟩
\end{pmatrix}
```

and

```math
e^{-i G̃ dt} \begin{pmatrix} 0 \\ \vdots \\ 0 \\ |Ψ⟩ \end{pmatrix}
= \begin{pmatrix}
\frac{∂}{∂ϵ₁} e^{-i Ĥ dt} |Ψ⟩ \\
\vdots \\
\frac{∂}{∂ϵₙ} e^{-i Ĥ dt} |Ψ⟩ \\
e^{-i Ĥ dt} |Ψ⟩
\end{pmatrix}.
```
"""
struct GradVector{num_controls,T}
    state::T
    grad_states::Vector{T}
end

function GradVector(Ψ::T, num_controls::Int64) where {T}
    grad_states = [similar(Ψ) for _ = 1:num_controls]
    for i = 1:num_controls
        fill!(grad_states[i], 0.0)
    end
    GradVector{num_controls,T}(copy(Ψ), grad_states)
end


"""Reset the given gradient vector for a new gradient evaluation.

```julia
resetgradvec!(Ψ̃::GradVector)
```

zeroes out `Ψ̃.grad_states` but leaves `Ψ̃.state` unaffected.

```julia
resetgradvec!(Ψ̃::GradVector, Ψ)
```

additionally sets `Ψ̃.state` to `Ψ`.
"""
function resetgradvec!(Ψ̃::GradVector)
    for i = 1:length(Ψ̃.grad_states)
        fill!(Ψ̃.grad_states[i], 0.0)
    end
end

function resetgradvec!(Ψ̃::GradVector{num_controls,T}, Ψ::T) where {num_controls,T}
    copyto!(Ψ̃.state, Ψ)
    resetgradvec!(Ψ̃)
end


function LinearAlgebra.mul!(Φ::GradVector, G::GradgenOperator, Ψ::GradVector)
    LinearAlgebra.mul!(Φ.state, G.G, Ψ.state)
    for i = 1:length(Ψ.grad_states)
        LinearAlgebra.mul!(Φ.grad_states[i], G.G, Ψ.grad_states[i])
        LinearAlgebra.mul!(Φ.grad_states[i], G.control_deriv_ops[i], Ψ.state, 1, 1)
    end
end


function LinearAlgebra.lmul!(c, Ψ::GradVector)
    LinearAlgebra.lmul!(c, Ψ.state)
    for i ∈ eachindex(Ψ.grad_states)
        LinearAlgebra.lmul!(c, Ψ.grad_states[i])
    end
end

function LinearAlgebra.axpy!(a, X::GradVector, Y::GradVector)
    LinearAlgebra.axpy!(a, X.state, Y.state)
    for i ∈ eachindex(X.grad_states)
        LinearAlgebra.axpy!(a, X.grad_states[i], Y.grad_states[i])
    end
end


function LinearAlgebra.norm(Ψ::GradVector)
    nrm = LinearAlgebra.norm(Ψ.state)
    for i ∈ eachindex(Ψ.grad_states)
        nrm += LinearAlgebra.norm(Ψ.grad_states[i])
    end
    return nrm
end


function LinearAlgebra.dot(Ψ::GradVector, Φ::GradVector)
    c::ComplexF64 = LinearAlgebra.dot(Ψ.state, Φ.state)
    for i ∈ eachindex(Ψ.grad_states)
        c += LinearAlgebra.dot(Ψ.grad_states[i], Φ.grad_states[i])
    end
    return c
end


LinearAlgebra.ishermitian(G::GradgenOperator) = false


# Upper triangular block matrices have eigenvalues only from the diagonal
# blocks. This is an example for a matrix that has real eigenvalues despite not
# being Hermitian
QuantumPropagators.has_real_eigvals(G::GradgenOperator) =
    QuantumPropagators.has_real_eigvals(G.G)


function Base.isreal(G::GradgenOperator)
    return (isreal(G.G) && all(isreal(D for D in G.control_deriv_ops)))
end


function Base.copyto!(dest::GradVector, src::GradVector)
    copyto!(dest.state, src.state)
    for i = 1:length(src.grad_states)
        copyto!(dest.grad_states[i], src.grad_states[i])
    end
    return dest
end


function Base.copy(Ψ::GradVector)
    Φ = GradVector(Ψ.state, length(Ψ.grad_states))
    for i = 1:length(Ψ.grad_states)
        copyto!(Φ.grad_states[i], Ψ.grad_states[i])
    end
    return Φ
end


function Base.length(Ψ::GradVector)
    return length(Ψ.state) * (1 + length(Ψ.grad_states))
end


function Base.size(G::GradgenOperator)
    return Base.size(G.G)
end


function Base.similar(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return GradVector{num_controls,T}(similar(Ψ.state), [similar(ϕ) for ϕ ∈ Ψ.grad_states])
end

function Base.similar(G::GradgenOperator{num_controls,GT,CGT}) where {num_controls,GT,CGT}
    return GradgenOperator{num_controls,GT,CGT}(similar(G.G), similar(G.control_deriv_ops))
end


function Base.copyto!(dest::GradgenOperator, src::GradgenOperator)
    copyto!(dest.G, src.G)
    copyto!(dest.control_deriv_ops, src.control_deriv_ops)
end


function Base.fill!(Ψ::GradVector, v)
    Base.fill!(Ψ.state, v)
    for i = 1:length(Ψ.grad_states)
        Base.fill!(Ψ.grad_states[i], v)
    end
end


function -(Ψ::GradVector, Φ::GradVector)
    res = copy(Ψ)
    LinearAlgebra.axpy!(-1, Φ.state, res.state)
    for i = 1:length(Ψ.grad_states)
        LinearAlgebra.axpy!(-1, Φ.grad_states[i], res.grad_states[i])
    end
    return res
end


function *(G::GradgenOperator{num_controls,GT,CGT}, α::Number) where {num_controls,GT,CGT}
    GradgenOperator{num_controls,GT,CGT}(G.G * α, [CG * α for CG in G.control_deriv_ops])
end

*(α::Number, G::GradgenOperator) = *(G::GradgenOperator, α::Number)


function QuantumPropagators.SpectralRange.random_state(H::GradgenOperator)
    state = QuantumPropagators.SpectralRange.random_state(H.G)
    num_controls = length(H.control_deriv_ops)
    grad_states = [
        QuantumPropagators.SpectralRange.random_state(H.G) for
        i ∈ eachindex(H.control_deriv_ops)
    ]
    return GradVector{num_controls,typeof(state)}(state, grad_states)
end


@inline function convert_gradgen_to_dense(G)
    N = size(G.G)[1]
    L = length(G.control_derivs)
    G_full = zeros(eltype(G.G), N * (L + 1), N * (L + 1))
    convert_gradgen_to_dense!(G_full, G)
end


@inline function convert_gradgen_to_dense!(G_full, G)
    N = size(G.G)[1]
    L = length(G.control_derivs)
    @inbounds for i = 1:L+1
        G_full[(i-1)*N+1:i*N, (i-1)*N+1:i*N] .= G.G
    end
    # Set the control-derivatives in the last (block-)column
    @inbounds for i = 1:L
        G_full[(i-1)*N+1:i*N, L*N+1:(L+1)*N] .= G.control_derivs[i]
    end
    return G_full
end


@inline function convert_gradvec_to_dense(Ψ)
    N = length(Ψ.state)
    L = length(Ψ.grad_states)
    Ψ_full = zeros(ComplexF64, N * (L + 1))
    convert_gradvec_to_dense!(Ψ_full, Ψ)
end


@inline function convert_gradvec_to_dense!(Ψ_full, Ψ)
    N = length(Ψ.state)
    L = length(Ψ.grad_states)
    @inbounds for i = 1:L
        Ψ_full[(i-1)*N+1:i*N] .= Ψ.grad_states[i]
    end
    @inbounds Ψ_full[L*N+1:(L+1)*N] .= Ψ.state
    return Ψ_full
end


@inline function convert_dense_to_gradvec!(Ψ, Ψ_full)
    N = length(Ψ.state)
    L = length(Ψ.grad_states)
    @inbounds for i = 1:L
        Ψ.grad_states[i] .= Ψ_full[(i-1)*N+1:i*N]
    end
    @inbounds Ψ.state .= Ψ_full[L*N+1:(L+1)*N]
    return Ψ
end

function Base.convert(::Type{Vector{ComplexF64}}, gradvec::GradVector)
    convert_gradvec_to_dense(gradvec)
end

function Base.convert(::Type{GradVector{num_controls,T}}, vec::T) where {num_controls,T}
    L = num_controls
    N = length(vec) ÷ (L + 1)  # dimension of state
    @assert length(vec) == (L + 1) * N
    grad_states = [vec[(i-1)*N+1:i*N] for i = 1:L]
    state = vec[L*N+1:(L+1)*N]
    return GradVector{num_controls,T}(state, grad_states)
end


function Base.Array{T}(G::GradgenOperator) where {T}
    N, M = size(G.G)
    L = length(G.control_deriv_ops)
    𝟘 = zeros(T, N, M)
    μ = G.control_deriv_ops
    block_rows = [
        hcat([𝟘 for j = 1:i-1]..., Array{T}(G.G), [𝟘 for j = i+1:L]..., Array{T}(μ[i]))
        for i = 1:L
    ]
    last_block_row = hcat([𝟘 for j = 1:L]..., Array{T}(G.G))
    return Base.Array{T}(vcat(block_rows..., last_block_row))
end

Base.Array(G::GradgenOperator) = Array{ComplexF64}(G)

function Base.convert(::Type{MT}, G::GradgenOperator) where {MT<:Matrix}
    Base.convert(MT, Base.Array(G))
end

QuantumPropagators._exp_prop_convert_state(::GradVector) = Vector{ComplexF64}
QuantumPropagators._exp_prop_convert_operator(::GradgenOperator) = Matrix{ComplexF64}
QuantumPropagators._exp_prop_convert_operator(::GradGenerator) = Matrix{ComplexF64}
