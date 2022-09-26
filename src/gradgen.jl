import LinearAlgebra
import QuantumPropagators
import Base: -, *

using QuantumPropagators.Controls: getcontrols, getcontrolderivs


@doc raw"""Extended generator for the standard dynamic gradient.

```julia
G̃ = TimeDependentGradGenerator(G)
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

Note that the ``∂G/∂ϵₗ(t)`` (``Ĥₗ`` in the above example) are functions, to
account for the possibility of non-linear control terms, see
[`getcontrolderiv`](@ref QuantumPropagators.Controls.getcontrolderiv).
"""
struct TimeDependentGradGenerator{GT,CT}
    G::GT
    control_derivs::Vector{Function}
    controls::Vector{CT}

    function TimeDependentGradGenerator(G::GT) where {GT}
        controls = collect(getcontrols(G))
        control_derivs = getcontrolderivs(G, controls)
        new{GT,eltype(controls)}(G, control_derivs, controls)
    end

end


"""Static generator for the standard dynamic gradient.

```julia
G::GradGenerator = evalcontrols(G_of_t::TimeDependentGradGenerator, vals_dict)
```

is the result of plugging in specific values for all controls in a
[`TimeDependentGradGenerator`](@ref). See [`evalcontrols`](@ref
QuantumPropagators.Controls.evalcontrols) and [`evalcontrols!`](@ref
QuantumPropagators.Controls.evalcontrols!).

The resulting object can be multiplied directly with a [`GradVector`](@ref),
e.g., in the process of evaluating a piecewise-constant time propagation.
"""
struct GradGenerator{num_controls,GT,CGT}
    G::GT
    control_derivs::Vector{CGT}
end

# Dummy initializer: this creates a GradGenerator that fits a
# TimeDependentGradGenerator structurally
function GradGenerator(G_of_t::TimeDependentGradGenerator)
    dummy_vals = IdDict(control => 1.0 for control in G_of_t.controls)
    dummy_tlist = [0.0, 1.0]
    G = QuantumPropagators.Controls.evalcontrols(G_of_t.G, dummy_vals, dummy_tlist, 1)
    control_derivs = [μ(1.0, dummy_tlist, 1) for μ in G_of_t.control_derivs]
    num_controls = length(control_derivs)
    GradGenerator{num_controls,typeof(G),eltype(control_derivs)}(G, control_derivs)
end


function QuantumPropagators.Controls.getcontrols(G_of_t::TimeDependentGradGenerator)
    return getcontrols(G_of_t.G)
end


function QuantumPropagators.Controls.evalcontrols!(
    G::GradGenerator,
    G_of_t::TimeDependentGradGenerator,
    vals_dict::AbstractDict,
    args...
)
    QuantumPropagators.Controls.evalcontrols!(G.G, G_of_t.G, vals_dict, args...)
    for (i, control) in enumerate(G_of_t.controls)
        μ = G_of_t.control_derivs[i]
        G.control_derivs[i] = μ(vals_dict[control], args...)
        # In most cases (for linear controls), the above line will be a no-op.
        # Hence, we're not using `copyto!`.
    end
    return G
end


function QuantumPropagators.Controls.evalcontrols(
    G_of_t::TimeDependentGradGenerator,
    vals_dict::AbstractDict,
    args...
)
    G = GradGenerator(G_of_t)
    QuantumPropagators.Controls.evalcontrols!(G, G_of_t, vals_dict, args...)
end


@doc raw"""Extended state-vector for the dynamic gradient.

```julia
Ψ̃ = GradVector(Ψ, num_controls)
```

for an initial state `Ψ` and `num_controls` control fields.

The `GradVector` conceptually corresponds to a direct-sum (block) column-vector
``Ψ̃ = (|Ψ̃₁⟩, |Ψ̃₂⟩, … |Ψ̃ₙ⟩, |Ψ⟩)^T``, where ``n`` is `num_controls`. With a
matching ``G̃`` as in the documentation of [`TimeDependentGradGenerator`](@ref),
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


function LinearAlgebra.mul!(Φ::GradVector, G::GradGenerator, Ψ::GradVector)
    LinearAlgebra.mul!(Φ.state, G.G, Ψ.state)
    for i = 1:length(Ψ.grad_states)
        LinearAlgebra.mul!(Φ.grad_states[i], G.G, Ψ.grad_states[i])
        LinearAlgebra.mul!(Φ.grad_states[i], G.control_derivs[i], Ψ.state, 1, 1)
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


LinearAlgebra.ishermitian(G::GradGenerator) = false


# Upper triangular block matrices have eigenvalues only from the diagonal
# blocks. This is an example for a matrix that has real eigenvalues despite not
# being Hermitian
QuantumPropagators.has_real_eigvals(G::GradGenerator) =
    QuantumPropagators.has_real_eigvals(G.G)


function Base.isreal(G::GradGenerator)
    return (isreal(G.G) && all(isreal(D for D in G.control_derivs)))
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


function Base.size(G::GradGenerator)
    return Base.size(G.G)
end


function Base.similar(Ψ::GradVector{num_controls,T}) where {num_controls,T}
    return GradVector{num_controls,T}(similar(Ψ.state), [similar(ϕ) for ϕ ∈ Ψ.grad_states])
end

function Base.similar(G::GradGenerator{num_controls,GT,CGT}) where {num_controls,GT,CGT}
    return GradGenerator{num_controls,GT,CGT}(similar(G.G), similar(G.control_derivs))
end


function Base.copyto!(dest::GradGenerator, src::GradGenerator)
    copyto!(dest.G, src.G)
    copyto!(dest.control_derivs, src.control_derivs)
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


function *(G::GradGenerator{num_controls,GT,CGT}, α::Number) where {num_controls,GT,CGT}
    GradGenerator{num_controls,GT,CGT}(G.G * α, [CG * α for CG in G.control_derivs])
end

*(α::Number, G::GradGenerator) = *(G::GradGenerator, α::Number)


function QuantumPropagators.SpectralRange.random_state(H::GradGenerator)
    state = QuantumPropagators.SpectralRange.random_state(H.G)
    num_controls = length(H.control_derivs)
    grad_states = [
        QuantumPropagators.SpectralRange.random_state(H.G) for
        i ∈ eachindex(H.control_derivs)
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


function Base.Array(G::GradGenerator)
    N, M = size(G.G)
    L = length(G.control_derivs)
    𝟘 = zeros(eltype(G.G), N, M)
    μ = G.control_derivs
    block_rows = [hcat([𝟘 for j = 1:i-1]..., G.G, [𝟘 for j = i+1:L]..., μ[i]) for i = 1:L]
    last_block_row = hcat([𝟘 for j = 1:L]..., G.G)
    return Base.Array(vcat(block_rows..., last_block_row))
end

function Base.convert(::Type{MT}, G::GradGenerator) where {MT<:Matrix}
    Base.convert(MT, Base.Array(G))
end

QuantumPropagators._exp_prop_convert_state(::GradVector) = Vector{ComplexF64}
QuantumPropagators._exp_prop_convert_operator(::GradGenerator) = Matrix{ComplexF64}
QuantumPropagators._exp_prop_convert_operator(::TimeDependentGradGenerator) =
    Matrix{ComplexF64}
