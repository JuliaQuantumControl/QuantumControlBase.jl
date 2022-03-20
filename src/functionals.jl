module Functionals

export J_T_ss, J_T_sm, J_T_re
export gate_functional, make_gate_chi
export make_gradient, make_chi

import ..WeightedObjective


using LinearAlgebra
using Zygote: Zygote
using FiniteDifferences: FiniteDifferences


@doc raw"""
Average complex overlap of the target states with forward-propagated states.

```julia
f_tau(ϕ, objectives; τ=nothing)
```

calculates

```math
f_τ = \frac{1}{N} \sum_{k=1}^{N} w_k τ_k
```

with

```math
τ_k = ⟨ϕ_k^\tgt|ϕ_k(T)⟩
```

in Hilbert space, or

```math
τ_k = \tr[ρ̂_k^{\tgt\,\dagger} ρ̂_k(T)]
```

in Liouville space, where ``|ϕ_k⟩`` or ``ρ̂_k`` are the elements
of `ϕ`, and ``|ϕ_k^\tgt⟩`` or ``ρ̂_k^\tgt`` are the
target states from the `target_state` field of the `objectives`. If `τ` is
given as a keyword argument, it must contain the values `τ_k` according to the
above definition. Otherwise, the ``τ_k`` values will be calculated internally.

``N`` is the number of objectives, and ``w_k`` is an optional weight for each
objective. For any objective that has a `weight` attribute (cf.
[`WeightedObjective`](@ref)), the ``w_k`` is taken from that attribute;
otherwise, ``w_k = 1``. The weights, if present, are not automatically
normalized, they are assumed to have values such that the resulting ``f_τ``
lies in the unit circle of the complex plane. Usually, this means that the
weights should sum to ``N``.
"""
function f_tau(ϕ, objectives; τ=nothing)
    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k = 1:N]
    end
    f::ComplexF64 = 0
    for k = 1:N
        obj = objectives[k]
        w = isa(obj, WeightedObjective) ? obj.weight : 1.0
        f += w * τ[k]
    end
    return f / N
end


@doc raw"""State-to-state phase-insensitive fidelity.

```julia
F_ss(ϕ, objectives; τ=nothing)
```

calculates

```math
F_{\text{ss}} = \frac{1}{N} \sum_{k=1}^{N} w_k |τ_k|^2 \quad\in [0, 1]
```

with ``N``, ``w_k`` and ``τ_k`` as in [`f_tau`](@ref).
"""
function F_ss(ϕ, objectives; τ=nothing)
    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k = 1:N]
    end
    F::ComplexF64 = f_tau(ϕ, objectives; τ=abs.(τ) .^ 2)
    @assert imag(F) < 1e-10
    return real(F)
end

@doc raw"""State-to-state phase-insensitive functional.

```julia
J_T_ss(ϕ, objectives; τ=nothing)
```

calculates

```math
J_{T,\text{ss}} = 1 - F_{\text{ss}} \in [0, 1].
```

All arguments are passed to [`F_ss`](@ref).
"""
function J_T_ss(ϕ, objectives; τ=nothing)
    return 1.0 - F_ss(ϕ, objectives; τ=τ)
end


@doc raw"""Krotov-states χ for functional [`J_T_ss`](@ref).

```julia
chi_ss!(χ, ϕ, objectives; τ=nothing)
```

sets the elements of `χ` according to

```math
|χ_k⟩
= -\frac{∂ J_{T,\text{ss}}}{∂ ⟨ϕ_k(T)|}
= \frac{1}{N} w_k τ_k |ϕ^{\tgt}_k⟩
```

with ``|ϕ^{\tgt}_k⟩``, ``τ_k`` and ``w_k`` as defined in [`f_tau`](@ref).

Note: this function can be obtained with `make_chi(J_T_ss, objectives)`.
"""
function chi_ss!(χ, ϕ, objectives; τ=nothing)
    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k = 1:N]
    end
    for k = 1:N
        obj = objectives[k]
        ϕₖ_tgt = obj.target_state
        copyto!(χ[k], ϕₖ_tgt)
        w = isa(obj, WeightedObjective) ? obj.weight : 1.0
        lmul!((τ[k] * w) / N, χ[k])
    end
end


@doc raw"""Square-modulus fidelity.

```julia
F_sm(ϕ, objectives; τ=nothing)
```

calculates

```math
F_{\text{sm}} = |f_τ|^2  \quad\in [0, 1].
```

All arguments are passed to [`f_tau`](@ref) to evaluate ``f_τ``.
"""
function F_sm(ϕ, objectives; τ=nothing)
    return abs(f_tau(ϕ, objectives; τ=τ))^2
end


@doc raw"""Square-modulus functional.

```julia
J_T_sm(ϕ, objectives; τ=nothing)
```

calculates

```math
J_{T,\text{sm}} = 1 - F_{\text{sm}} \quad\in [0, 1].
```

All arguments are passed to [`f_tau`](@ref) while evaluating ``F_{\text{sm}}``
in [`F_sm`](@ref).
"""
function J_T_sm(ϕ, objectives; τ=nothing)
    return 1.0 - F_sm(ϕ, objectives; τ=τ)
end


@doc raw"""Gradient for [`J_T_sm`](@ref).

```julia
grad_J_T_sm!(∇J_T, τ, ∇τ)
```

analytically sets the elements of the gradient `∇J_T` according to

```math
\frac{∂ J_{T,\text{sm}}(\{τ_k\})}{\partial ϵ_{ln}}
= \frac{1}{N^2} \sum_{k=1}^N \sum_{k'=1}^N\left[
        \frac{\partial τ_{k'}^*}{∂ϵ_{ln}} τ_k +
        τ_{k'}^* \frac{\partial τ_k^*}{∂ϵ_{ln}}
   \right]
= -\frac{2}{N} \Re \sum_{k=1}^N \sum_{k'=1}^N
  τ_{k'}^* \frac{∂τ_k}{\partial ϵ_{ln}}
```

with all quantities as defined in [`make_gradient`](@ref).

Note: this function can be obtained with
`make_gradient(J_T_sm, objectives, via=:tau)`.
"""
function grad_J_T_sm!(∇J_T, τ, ∇τ)
    N = length(τ) # number of objectives
    L, N_T = size(∇τ[1])  # number of controls/time intervals
    ∇J_T′ = reshape(∇J_T, L, N_T)  # writing to ∇J_T′ modifies ∇J_T
    for l = 1:L
        for n = 1:N_T
            ∇J_T′[l, n] = real(sum([conj(τ[k′]) * ∇τ[k][l, n] for k′ = 1:N for k = 1:N]))
        end
    end
    lmul!(-2 / N^2, ∇J_T)
    return ∇J_T
end


@doc raw"""Krotov-states χ for functional [`J_T_sm`](@ref).

```julia
chi_sm!(χ, ϕ, objectives; τ=nothing)
```

sets the elements of `χ` according to

```math
|χ_k⟩
= -\frac{\partial J_{T,\text{sm}}}{\partial ⟨ϕ_k(T)|}
= \frac{1}{N^2} w_k \sum_{j}^{N} w_j τ_j |ϕ_k^{\tgt}⟩
```

with ``|ϕ^{\tgt}_k⟩``, ``τ_j`` and ``w_k`` as defined in [`f_tau`](@ref).

Note: this function can be obtained with `make_chi(J_T_sm, objectives)`.
"""
function chi_sm!(χ, ϕ, objectives; τ=nothing)

    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k = 1:N]
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


@doc raw"""Real-part fidelity.

```julia
F_re(ϕ, objectives; τ=nothing)
```

calculates

```math
F_{\text{re}} = \Re[f_{τ}] \quad\in \begin{cases}
    [-1, 1] & \text{in Hilbert space} \\
    [0, 1] & \text{in Liouville space.}
\end{cases}
```

All arguments are passed to [`f_tau`](@ref) to evaluate ``f_τ``.
"""
function F_re(ϕ, objectives; τ=nothing)
    return real(f_tau(ϕ, objectives; τ=τ))
end


@doc raw"""Real-part functional.

```julia
J_T_re(ϕ, objectives; τ=nothing)
```

calculates

```math
J_{T,\text{re}} = 1 - F_{\text{re}} \quad\in \begin{cases}
    [0, 2] & \text{in Hilbert space} \\
    [0, 1] & \text{in Liouville space.}
\end{cases}
```

All arguments are passed to [`f_tau`](@ref) while evaluating ``F_{\text{re}}``
in [`F_re`](@ref).
"""
function J_T_re(ϕ, objectives; τ=nothing)
    return 1.0 - F_re(ϕ, objectives; τ=τ)
end


@doc raw"""Krotov-states χ for functional [`J_T_re`](@ref).

```julia
chi_re!(χ, ϕ, objectives; τ=nothing)
```

sets the elements of `χ` according to


```math
|χ_k⟩
= -\frac{∂ J_{T,\text{re}}}{∂ ⟨ϕ_k(T)|}
= \frac{1}{2N} w_k |ϕ^{\tgt}_k⟩
```

with ``|ϕ^{\tgt}_k⟩`` and ``w_k`` as defined in [`f_tau`](@ref).

Note: this function can be obtained with `make_chi(J_T_re, objectives)`.
"""
function chi_re!(χ, ϕ, objectives; τ=nothing)
    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k = 1:N]
    end
    for k = 1:N
        obj = objectives[k]
        ϕₖ_tgt = obj.target_state
        copyto!(χ[k], ϕₖ_tgt)
        w = isa(obj, WeightedObjective) ? obj.weight : 1.0
        lmul!(w / (2N), χ[k])
    end
end




"""Convert a functional from acting on a gate to acting on propagated states.

```
J_T = gate_functional(J_T_U; kwargs...)
```

constructs a functional `J_T` that meets the requirements for
[`make_gradient`](@ref) and [`make_chi`](@ref). That is, the output `J_T` takes
positional positional arguments `ϕ` and `objectives`. The input functional
`J_T_U` is assumed to have the signature `J_T_U(U; kwargs...)` where `U` is a
matrix with elements ``U_{ij} = ⟨Ψ_i|ϕ_j⟩``, where ``|Ψ_i⟩`` is the
`initial_state` of the i'th `objectives` (assumed to be the i'th canonical
basis state) and ``|ϕ_j⟩`` is the result of forward-propagating ``|Ψ_j⟩``. That
is, `U` is the projection of the time evolution operator into the subspace
defined by the basis in the `initial_states` of the  `objectives`.
"""
function gate_functional(J_T_U; kwargs...)

    function J_T(ϕ, objectives; τ=nothing)
        N = length(objectives)
        U = [(objectives[i].initial_state ⋅ ϕ[j]) for i = 1:N, j = 1:N]
        return J_T_U(U; kwargs...)
    end

    return J_T

end


@doc raw"""
Return a function to evaluate ``|χ_k⟩ = -∂J_T(Û)/∂⟨ϕ_k|`` via the chain rule.

```julia
chi! = make_gate_chi(J_T_U, objectives; use_finite_differences=false, kwargs...)
```

returns a function equivalent to

```julia
chi! = make_chi(gate_functional(J_T_U; kwargs...), objectives)
```

```math
\begin{split}
    |χ_k⟩
    &= -\frac{∂}{∂⟨ϕ_k|} J_T \\
    &= - \frac{1}{2} \sum_i (∇_U J_T)_{ik} \frac{∂ U_{ik}}{∂⟨ϕ_k|} \\
    &= - \frac{1}{2} \sum_i (∇_U J_T)_{ik} |Ψ_i⟩
\end{split}
```

where ``|Ψ_i⟩`` is the basis state stored as the `initial_state` of the i'th
`objective`, see [`gate_functional`](@ref).

The gradient ``∇_U J_T`` is obtained via automatic differentiation, or via
finite differences if `use_finite_differences=true`.

Compared to the more general [`make_chi`](@ref), `make_gate_chi` will generally
have a slightly smaller numerical overhead, as it pushes the use of automatic
differentiation down by one level.

With `use_finite_differences=true`, this routine serves to test and debug
gradients for gate functionals obtained by automatic differentiation.
"""
function make_gate_chi(J_T_U, objectives; use_finite_differences=false, kwargs...)

    N = length(objectives)
    basis = [obj.initial_state for obj in objectives]

    function zygote_gate_chi!(χ, ϕ, objectives; τ=nothing)
        function _J_T(U)
            -J_T_U(U; kwargs...)
        end
        U = [basis[i] ⋅ ϕ[j] for i = 1:N, j = 1:N]
        if use_finite_differences
            fdm = FiniteDifferences.central_fdm(5, 1)
            ∇J = FiniteDifferences.grad(fdm, gate -> _J_T(gate), U)[1]
        else
            ∇J = Zygote.gradient(gate -> _J_T(gate), U)[1]
        end
        for k = 1:N
            χ[k] .= 0.5 * sum([∇J[i, k] * basis[i] for i = 1:N])
        end
    end

    return zygote_gate_chi!

end


@doc raw"""Gradient for an arbitrary functional evaluated via χ-states.

```julia
grad_J_T_via_chi!(∇J_T, τ, ∇τ)
```

sets the (vectorized) elements of the gradient `∇J_T` to the gradient
``∂J_T/∂ϵ_{ln}`` for an arbitrary functional ``J_T=J_T(\{|ϕ_k(T)⟩\})``, under
the assumption that

```math
\begin{aligned}
    τ_k &= ⟨χ_k|ϕ_k(T)⟩ \quad \text{with} \quad |χ_k⟩ &= -∂J_T/∂⟨ϕ_k(T)|
    \quad \text{and} \\
    ∇τ_{kln} &= ∂τ_k/∂ϵ_{ln}\,,
\end{aligned}
```

where ``|ϕ_k(T)⟩`` is a state resulting from the forward propagation of some
initial state ``|ϕ_k⟩`` under the pulse values ``ϵ_{ln}`` where ``l`` numbers
the controls and ``n`` numbers the time slices. The ``τ_k`` are the elements of
`τ` and ``∇τ_{kln}`` corresponds to `∇τ[k][l, n]`.

In this case,

```math
(∇J_T)_{ln} = ∂J_T/∂ϵ_{ln} = -2 \Re \sum_k ∇τ_{kln}\,,
```

see [`make_gradient`](@ref).

Note that the definition of the ``|χ_k⟩`` matches exactly the definition
of the boundary condition for the backward propagation in Krotov's method, see
[`make_chi`](@ref). Specifically, there is a minus sign in front of the
derivative, compensated by the minus sign in the factor ``(-2)`` of the final
``(∇J_T)_{ln}``.
"""
function grad_J_T_via_chi!(∇J_T, τ, ∇τ)
    N = length(τ) # number of objectives
    L, N_T = size(∇τ[1])  # number of controls/time intervals
    ∇J_T′ = reshape(∇J_T, L, N_T)  # writing to ∇J_T′ modifies ∇J_T
    for l = 1:L
        for n = 1:N_T
            ∇J_T′[l, n] = real(sum([∇τ[k][l, n] for k = 1:N]))
        end
    end
    lmul!(-2, ∇J_T)
    return ∇J_T
end


@doc raw"""Return a function that evaluates the gradient ``∇J_T``.

```julia
grad_func! = make_gradient(J_T, objectives; via=:tau, force_zygote=false)
```

creates a function `gradfunc!(∇J_T, τ, ∇τ)` that takes a vector `τ` of values
``τ_k`` and a vector `∇τ` of gradients ``∇τ_k`` where the ``(ln)``'th element
of ``∇τ_k`` is ``∂τ_k/∂ϵ_{ln}``, and writes the element ``(ln)`` of the
gradient `∇J_T` as ``∂J_T/∂ϵ_{ln}``. The definition of ``τ_k`` depends on
`via`, see below. The ``ϵ_{ln}`` are the values of the control control field
discretized to the midpoints of a time grid. The index ``l`` numbers the
control and ``n`` numbers the time slice. The gradient (like the controls) are
assumed to be vectorized. That is, ``∇J_T`` is a vector of values with a
double-index ``(ln)``.

The passed `J_T` parameter corresponding to the functional ``J_T`` must be a
function that takes a vector of forward-propagates states `ϕ` and a vector of
objectives as positional parameters.  It must also accept a vector `τ` as a
keyword argument, which contains the overlaps of the states in `ϕ` and the
`target_state` fields of the `objectives`. If the `objectices` do not define
`target_states`, or if the ``τ``-values are not available, `J_T` must accept
`τ=nothing`. See [`J_T_sm`](@ref) for an example.


## Gradient via τ

For `via=:tau` (default), we define

```math
τ_k ≡ ⟨ϕ_k^\tgt|ϕ_k(T)⟩
```

as the overlap of ``|ϕ_k(T)⟩`` resulting from the forward propagation of the
`initial_state` ``|ϕ_k⟩``  of the k'th objective under the pulse values
``ϵ_{ln}``, and ``|ϕ_k^\tgt⟩`` as the `target_state` of the k'th objective.

We then understand ``J_T`` as a function of the ``τ_k``, and evaluate the
elements of ``∇J_T`` via the chain rule:

```math
(∇J_T)_{ln} ≡ \frac{∂J_T(\{τ_k\})}{∂ϵ_{ln}}
= 2\Re\sum_k
    \frac{∂J_T}{∂τ_k}
    \frac{∂τ_k}{∂ϵ_{ln}}\,.
```

Since the ``τ_k`` are complex numbers,

```math
\frac{∂J_T}{∂τ_k} = \frac{1}{2}\left(
    \frac{∂ J_T}{∂ \Re[τ_k]}
    - i \frac{∂ J_T}{∂ \Im[τ_k]}
\right)
```

is defined as the [Wirtinger
derivative](https://www.ekinakyurek.me/complex-derivatives-wirtinger/), and

```math
\frac{∂τ_k}{∂ϵ_{ln}} = \frac{∂\Re[τ_k]}{∂ϵ_{ln}} + i \frac{∂\Im[τ_k]}{∂ϵ_{ln}}
```

is simply the derivative of a complex number with respect to the real-valued
``ϵ_{ln}``.

Thus, the returned `grad_func!` effectively encodes the outer derivative
``∂J_T/∂τ_k``. For functionals where that derivative is known analytically,
the analytic expression is used, e.g., [`J_T_sm`](@ref) →
[`grad_J_T_sm!`](@ref).

Otherwise, or if `force_zygote=true`, the outer derivative is
determined directly from `J_T`, via automatic differentiation (using
[Zygote](https://fluxml.ai/Zygote.jl)).

## Gradient via χ

For `via=:chi`, the functional ``J_T`` is understood directly as a function of
the forward-propagated states ``|ϕ_k⟩`` instead of a function of overlaps with
the target states. This is useful in particular if the `objectives` do not
define objectives and/or the functional `J_T` cannot be expressed in terms of
overlaps.

Again we apply a chain rule to calculate the elements of the gradient ``∇J_T``:

```math
\begin{split}
(∇J_T)_{ln} &≡ \frac{∂J_T(\{|ϕ_k(T)⟩\})}{∂ϵ_{ln}}\\
&= 2\Re\sum_k
    \frac{∂J_T}{∂|ϕ_k(T)⟩}
    \frac{∂|ϕ_k(T)⟩}{∂ϵ_{ln}} \\
&= -2 \Re \sum_k \frac{∂}{∂ϵ_{ln}} ⟨χ_k(T)|ϕ_k(T)⟩\,,
\end{split}
```

with

```math
|χ_k⟩
= -\frac{J_T}{⟨ϕ_k(T)|}
= -\frac{1}{2}\left(
    \left\vert \frac{∂J_T}{∂\Re[ϕ_k]} \right\rangle
    + i \left\vert \frac{∂J_T}{∂\Im[ϕ_k]} \right\rangle
    \right)
```

as a matrix-calculus extension of the Wirtinger derivative. This definition of
``|χ_k⟩`` (note the minus sign!) matches the definition of the boundary
condition in Krotov's method, and for a given functional `J_T`, the
states ``|χ_k⟩`` can be obtained with [`make_chi`](@ref).

We define

```math
τ_k ≡ ⟨χ_k(T)|ϕ_k(T)⟩
```

and associate the `gradfunc!` argument `∇τ[k][l, n]` with

```math
(∇τ_k)_{ln} = \frac{∂τ_k}{∂ ϵ_{ln}}
```

so that structurally, ``(∇J_T)_{ln}`` is the same as for `via=:tau`, just that
``τ_k`` is now defined with respect to the boundary condition state ``|χ_k⟩``
instead of the target state ``|ϕ_k^\tgt⟩``.

The returned `grad_func!` that encodes the above equations is
[`grad_J_T_via_chi!`](@ref). This is independent of `J_T`, since the dependency
on the functional `J_T` is entirely encoded in the states ``|χ_k(T)⟩``, and
thus the gradient `∇τ`. Also, `force_zygote=true` has no effect for `via=:chi`.
Instead, `force_zygote` should be passed to the underlying [`make_chi`](@ref).

!!! tip

    In order to extend `make_gradient` with an analytic implementation for a
    new `J_T` function, define a new method like so:

    ```julia
    make_gradient(::typeof(J_T_sm), objectives, via::Val{:tau}) = grad_J_T_sm!
    ```

    which links `make_gradient` for [`J_T_sm`](@ref) to [`grad_J_T_sm!`](@ref).
"""
function make_gradient(J_T, objectives; via::Symbol=:tau, force_zygote=false)
    if force_zygote && (via == :tau)
        return make_zygote_gradient(J_T, objectives)
    else
        return make_gradient(J_T, objectives, Val(via))
    end
end

function make_zygote_gradient(J_T, objectives)

    function zygote_gradfunc!(∇J_T, τ, ∇τ)
        ∇J = Zygote.gradient(τ -> J_T(nothing, objectives; τ=τ), τ)[1]
        ∇J_T .= vec(sum(real(∇J) .* real.(∇τ) + imag(∇J) .* imag.(∇τ)))
    end

    return zygote_gradfunc!
end

make_gradient(J_T, objectives, via::Val{:chi}) = grad_J_T_via_chi!
make_gradient(J_T, objectives, via::Val{:tau}) = make_zygote_gradient(J_T, objectives)
make_gradient(::typeof(J_T_sm), objectives, via::Val{:tau}) = grad_J_T_sm!


@doc raw"""Return a function that evaluates ``|χ_k⟩ = -∂J_T/∂⟨ϕ_k|``.

```julia
chi! = make_chi(J_T, objectives; force_zygote=false)
```

creates a function `chi!(χ, ϕ, objectives; τ=nothing)` that sets
``|χ_k⟩ = -∂J_T/∂⟨ϕ_k|``. This is the state used as the boundary condition for
the backward propagation propagation in Krotov's method, as well as GRAPE if
[`grad_J_T_via_chi!`](@ref) is used. It is defined as a [Wirtinger
derivative](https://www.ekinakyurek.me/complex-derivatives-wirtinger/),
see [`make_gradient`](@ref).

The function `J_T` must take a vector of states `ϕ`
and a vector of `objectives` as positional parameters, and a vector `τ` as a
keyword argument, see e.g. [`J_T_sm`](@ref). If all objectives define a
`target_state`, then `τ` will be the overlap of the states `ϕ` with those
target states. The functional `J_T` may or may not use those overlaps.
Likewise, the resulting `chi!` may or may not use the keyword parameter `τ`.

For functionals where ``-∂J_T/∂⟨ϕ_k|`` is known analytically, that analytic
derivative will be returned, e.g.,

* [`J_T_sm`](@ref) → [`chi_sm!`](@ref),
* [`J_T_re`](@ref) → [`chi_re!`](@ref),
* [`J_T_ss`](@ref) → [`chi_ss!`](@ref).

Otherwise, or if `force_zygote=true`, automatic differentiation via Zygote is
used to calculate the derivative directly from `J_T`.

!!! tip

    In order to extend `make_chi` with an analytic implementation for a new
    `J_T` function, define a new method `make_analytic_chi` like so:

    ```julia
    make_analytic_chi(::typeof(J_T_sm), objectives) = chi_sm!
    ```

    which links `make_chi` for [`J_T_sm`](@ref) to [`chi_sm!`](@ref).
"""
function make_chi(J_T, objectives; force_zygote=false)
    if force_zygote
        return make_zygote_chi(J_T, objectives)
    else
        return make_analytic_chi(J_T, objectives)
    end
end

function make_zygote_chi(J_T, objectives)

    N = length(objectives)

    function zygote_chi!(χ, ϕ, objectives; τ=nothing)
        function _J_T(Ψ...)
            -J_T(Ψ, objectives)
        end
        for (k, ∇J) ∈ enumerate(Zygote.gradient(_J_T, ϕ...))
            copyto!(χ[k], ∇J)
            lmul!(0.5, χ[k])
        end
    end

    return zygote_chi!

end

make_analytic_chi(J_T, objectives) = make_zygote_chi(J_T, objectives)
# Well, the above fallback to `make_zygote_chi` is clearly not "analytic", but
# the intent of this name is to allow users to define new analytic
# implementation, cf. the explanation in the doc of `make_chi`.
make_analytic_chi(::typeof(J_T_sm), objectives) = chi_sm!
make_analytic_chi(::typeof(J_T_re), objectives) = chi_re!
make_analytic_chi(::typeof(J_T_ss), objectives) = chi_ss!


end
