import LinearAlgebra
import QuantumPropagators
import Base: -, *

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
[`getcontrolderiv`](@ref).
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
[`TimeDependentGradGenerator`](@ref). See [`evalcontrols`](@ref) and
[`evalcontrols!`](@ref).

The resulting object can be multiplied directly with a [`GradVector`](@ref),
e.g., in the process of evaluating a piecewise-constant time propagation.
"""
struct GradGenerator{GT,CGT}
    G::GT
    control_derivs::Vector{CGT}
end

function GradGenerator(G_of_t::TimeDependentGradGenerator)
    dummy_vals = IdDict(control => 1.0 for control in G_of_t.controls)
    G = evalcontrols(G_of_t.G, dummy_vals)
    control_derivs = [μ(1.0) for μ in G_of_t.control_derivs]
    GradGenerator{typeof(G),eltype(control_derivs)}(G, control_derivs)
end


function evalcontrols!(
    G::GradGenerator,
    G_of_t::TimeDependentGradGenerator,
    vals_dict::AbstractDict
)
    evalcontrols!(G.G, G_of_t.G, vals_dict)
    for (i, control) in enumerate(G_of_t.controls)
        G.control_derivs[i] = G_of_t.control_derivs[i](vals_dict[control])
        # In most cases (for linear controls), the above line will be a no-op.
        # Hence, we're not using `copyto!`.
    end
    return G
end


function evalcontrols(G_of_t::TimeDependentGradGenerator, vals_dict::AbstractDict)
    G = GradGenerator(G_of_t)
    evalcontrols!(G, G_of_t, vals_dict)
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
struct GradVector{T}
    state::T
    grad_states::Vector{T}
end

function GradVector(Ψ::T, num_controls::Int64) where {T}
    grad_states = [similar(Ψ) for _ = 1:num_controls]
    for i = 1:num_controls
        fill!(grad_states[i], 0.0)
    end
    GradVector{T}(copy(Ψ), grad_states)
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
function resetgradvec!(Ψ̃::GradVector{T}) where {T}
    for i = 1:length(Ψ̃.grad_states)
        fill!(Ψ̃.grad_states[i], 0.0)
    end
end

function resetgradvec!(Ψ̃::GradVector{T}, Ψ::T) where {T}
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
    for i = 1:length(Ψ.grad_states)
        LinearAlgebra.lmul!(c, Ψ.grad_states[i])
    end
end

function LinearAlgebra.axpy!(a, X::GradVector, Y::GradVector)
    LinearAlgebra.axpy!(a, X.state, Y.state)
    for i = 1:length(X.grad_states)
        LinearAlgebra.axpy!(a, X.grad_states[i], Y.grad_states[i])
    end
end


function LinearAlgebra.norm(Ψ::GradVector)
    nrm = LinearAlgebra.norm(Ψ.state)
    for i = 1:length(Ψ.grad_states)
        nrm += LinearAlgebra.norm(Ψ.grad_states[i])
    end
    return nrm
end


function LinearAlgebra.dot(Ψ::GradVector, Φ::GradVector)
    c::ComplexF64 = LinearAlgebra.dot(Ψ.state, Φ.state)
    for i = 1:length(Ψ.grad_states)
        c += LinearAlgebra.dot(Ψ.grad_states[i], Φ.grad_states[i])
    end
    return c
end


LinearAlgebra.ishermitian(G::GradGenerator) = false


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


function Base.similar(Ψ::GradVector)
    return GradVector(similar(Ψ.state), [similar(ϕ) for ϕ ∈ Ψ.grad_states])
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


function *(G::GradGenerator, α::Number)
    GradGenerator{typeof(G.G),eltype(G.control_derivs)}(
        G.G * α,
        [CG * α for CG in G.control_derivs]
    )
end

*(α::Number, G::GradGenerator) = *(G::GradGenerator, α::Number)


struct DenseGradExpPropWrk{T}
    Ψ_full::Vector{ComplexF64}
    Φ_full::Vector{ComplexF64}
    G_full::Matrix{T}
    function DenseGradExpPropWrk(
        Ψ̃::GradVector{Vector{ComplexF64}},
        G̃::GradGenerator{Matrix{T},Matrix{T}}
    ) where {T}
        Ψ_full = convert_gradvec_to_dense(Ψ̃)
        Φ_full = similar(Ψ_full)
        G_full = convert_gradgen_to_dense(G̃)
        new{T}(Ψ_full, Φ_full, G_full)
    end
end


function QuantumPropagators.initpropwrk(
    state::GradVector{Vector{ComplexF64}},
    tlist,
    method::Val{:expprop},
    generator::Vararg{GradGenerator{Matrix{T},Matrix{T}}};
    kwargs...
) where {T}
    return DenseGradExpPropWrk(state, generator[1])
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


function QuantumPropagators.propstep!(
    Ψ̃::GradVector{Vector{ComplexF64}},
    G̃::GradGenerator{Matrix{T},Matrix{T}},
    dt::Float64,
    wrk::DenseGradExpPropWrk;
    kwargs...
) where {T}
    func = get(kwargs, :func, H_dt -> exp(-1im * H_dt))
    convert_gradgen_to_dense!(wrk.G_full, G̃)
    convert_gradvec_to_dense!(wrk.Ψ_full, Ψ̃)
    U = func(wrk.G_full * dt)
    LinearAlgebra.mul!(wrk.Φ_full, U, wrk.Ψ_full)
    convert_dense_to_gradvec!(Ψ̃, wrk.Φ_full)
    return Ψ̃
end
