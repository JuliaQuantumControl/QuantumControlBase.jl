using LinearAlgebra

"""Average complex overlap of the target states with forward-propagated states.

```julia
f_tau(ϕ, objectives; τ=nothing)
```
"""
function f_tau(ϕ, objectives; τ=nothing)
    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k in 1:N]
    end
    f::ComplexF64 = 0
    for k = 1:N
        obj = objectives[k]
        w = isa(obj, WeightedObjective) ? obj.weight : 1.0
        f += w * τ[k]
    end
    return f / N
end


"""State-to-state phase-insensitive fidelity.

```julia
F_ss(ϕ, objectives; τ=nothing)
```
"""
function F_ss(ϕ, objectives; τ=nothing)
    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k in 1:N]
    end
    F::ComplexF64  = f_tau(ϕ, objectives; τ=abs.(τ).^2)
    @assert imag(F) < 1e-10
    return real(F)
end

"""State-to-state phase-insensitive functional.

```julia
J_T_ss(ϕ, objectives; τ=nothing)
```
"""
function J_T_ss(ϕ, objectives; τ=nothing)
    return 1.0 - F_ss(ϕ, objectives; τ=τ)
end


"""Krotov-states χ for functional [`J_T_ss`](@ref).

```julia
chi_ss!(χ, ϕ, objectives; τ=nothing)
```
"""
function chi_ss!(χ, ϕ, objectives; τ=nothing)
    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k in 1:N]
    end
    for k = 1:N
        obj = objectives[k]
        ϕₖ_tgt = obj.target_state
        copyto!(χ[k], ϕₖ_tgt)
        w = isa(obj, WeightedObjective) ? obj.weight : 1.0
        lmul!((τ[k] * w)/N, χ[k])
    end
end


"""Square-modulus fidelity.

```julia
F_sm(ϕ, objectives; τ=nothing)
```
"""
function F_sm(ϕ, objectives; τ=nothing)
    return abs(f_tau(ϕ, objectives; τ=τ))^2
end


"""Square-modulus functional.

```julia
J_T_sm(ϕ, objectives; τ=nothing)
```
"""
function J_T_sm(ϕ, objectives; τ=nothing)
    return 1.0 - F_sm(ϕ, objectives; τ=τ)
end


"""Gradient for [`J_T_sm`](@ref)."""
function grad_J_T_sm!(G, τ, ∇τ)
    N = length(τ) # number of objectives
    L, N_T = size(∇τ[1])  # number of controls/time intervals
    G′ = reshape(G, L, N_T)  # writing to G′ modifies G
    for l = 1:L
        for n = 1:N_T
            G′[l, n] = real(sum([conj(τ[k′]) * ∇τ[k][l, n]
                                 for k′=1:N for k=1:N]))
        end
    end
    lmul!(-2/N, G)
    return G
end


"""Krotov-states χ for functional [`J_T_sm`](@ref).

```julia
chi_sm!(χ, ϕ, objectives; τ=nothing)
```
"""
function chi_sm!(χ, ϕ, objectives; τ=nothing)

    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k in 1:N]
    end

    w = ones(N)
    for k = 1:N
        obj = objectives[k]
        w[k] = isa(obj, WeightedObjective) ? obj.weight : 1.0
    end

    for k = 1:N
        obj = objectives[k]
        ϕₖ_tgt = obj.target_state
        copyto!(χ[k], ϕₖ_tgt)
        lmul!(w[k] * sum(w .* τ) / N^2, χ[k])
    end

end


"""Real-part fidelity.

```julia
F_re(ϕ, objectives; τ=nothing)
```
"""
function F_re(ϕ, objectives; τ=nothing)
    return real(f_tau(ϕ, objectives; τ=τ))
end


"""Real-part functional.

```julia
J_T_re(ϕ, objectives; τ=nothing)
```
"""
function J_T_re(ϕ, objectives; τ=nothing)
    return 1.0 - F_re(ϕ, objectives; τ=τ)
end


"""Krotov-states χ for functional [`J_T_re`](@ref).

```julia
chi_re!(χ, ϕ, objectives; τ=nothing)
```
"""
function chi_re!(χ, ϕ, objectives; τ=nothing)
    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k in 1:N]
    end
    for k = 1:N
        obj = objectives[k]
        ϕₖ_tgt = obj.target_state
        copyto!(χ[k], ϕₖ_tgt)
        w = isa(obj, WeightedObjective) ? obj.weight : 1.0
        lmul!(w/(2N), χ[k])
    end
end
