import LinearAlgebra

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
struct TimeDependentGradGenerator{GT, CT}
    G :: GT
    control_derivs :: Vector{Function}
    controls :: Vector{CT}

    function TimeDependentGradGenerator(G::GT) where GT
        controls = collect(getcontrols(G))
        control_derivs = getcontrolderivs(G, controls)
        new{GT, eltype(controls)}(G, control_derivs, controls)
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
struct GradGenerator{GT, CGT}
    G :: GT
    control_derivs :: Vector{CGT}

    function GradGenerator(G_of_t::TimeDependentGradGenerator)
        dummy_vals = IdDict(control => 1.0  for control in G_of_t.controls)
        G = evalcontrols(G_of_t.G, dummy_vals)
        control_derivs = [μ(1.0) for μ in G_of_t.control_derivs]
        new{typeof(G), eltype(control_derivs)}(G, control_derivs)
    end
end


function evalcontrols!(G::GradGenerator, G_of_t::TimeDependentGradGenerator, vals_dict::AbstractDict)
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
    state :: T
    grad_states :: Vector{T}
end

function GradVector(Ψ::T, num_controls::Int64) where T
    grad_states = [similar(Ψ) for _ in 1:num_controls]
    for i = 1 : num_controls
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
function resetgradvec!(Ψ̃::GradVector{T}) where T
    for i = 1 : length(Ψ̃.grad_states)
        fill!(Ψ̃.grad_states[i], 0.0)
    end
end

function resetgradvec!(Ψ̃::GradVector{T}, Ψ::T) where T
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


function Base.similar(Ψ::GradVector)
    return GradVector(similar(Ψ.state), [similar(ϕ) for ϕ ∈ Ψ.grad_states])
end


function Base.fill!(Ψ::GradVector, v)
    Base.fill!(Ψ.state, v)
    for i = 1:length(Ψ.grad_states)
        Base.fill!(Ψ.grad_states[i], v)
    end
end
