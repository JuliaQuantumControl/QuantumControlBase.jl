module Functionals

export J_T_ss, J_T_sm, J_T_re
export gate_functional, make_gate_chi
export make_chi

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

``N`` is the number of objectives, and ``w_k`` is the `weight` attribute for
each objective. The weights are not automatically
normalized, they are assumed to have values such that the resulting ``f_τ``
lies in the unit circle of the complex plane. Usually, this means that the
weights should sum to ``N``.
"""
function f_tau(ϕ, objectives; τ=nothing)
    # TODO: keyword arguments should not use unicode
    N = length(objectives)
    if τ === nothing
        τ = [dot(objectives[k].target_state, ϕ[k]) for k = 1:N]
    end
    f::ComplexF64 = 0
    for k = 1:N
        obj = objectives[k]
        w = obj.weight
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
    f::Float64 = 0
    for k = 1:N
        obj = objectives[k]
        w = obj.weight
        f += w * abs2(τ[k])
    end
    return f / N
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


@doc raw"""Backward boundary states ``|χ⟩`` for functional [`J_T_ss`](@ref).

```julia
chi_ss!(χ, ϕ, objectives; τ=nothing)
```

sets the elements of `χ` according to

```math
|χ_k⟩
= -\frac{∂ J_{T,\text{ss}}}{∂ ⟨ϕ_k(T)|}
= \frac{1}{N} w_k τ_k |ϕ^{\tgt}_k⟩\,,
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
        w = obj.weight
        lmul!((τ[k] * w) / N, χ[k])
    end
end


@doc raw"""Square-modulus fidelity.

```julia
F_sm(ϕ, objectives; τ=nothing)
```

calculates

```math
F_{\text{sm}}
    = |f_τ|^2
    = \left\vert\frac{1}{N} \sum_{k=1}^{N} w_k τ_k\right\vert^2
    = \frac{1}{N^2} \sum_{k=1}^{N} \sum_{j=1}^{N} w_k w_j τ̄_k τ_j
    \quad\in [0, 1]\,,
```

with ``w_k`` the weight for the k'th objective and ``τ_k`` the overlap of the
k'th propagated state with the k'th target state, ``τ̄_k`` the complex conjugate
of ``τ_k``, and ``N`` the number of objectives.

All arguments are passed to [`f_tau`](@ref) to evaluate ``f_τ``.
"""
function F_sm(ϕ, objectives; τ=nothing)
    return abs2(f_tau(ϕ, objectives; τ=τ))
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


@doc raw"""Backward boundary states ``|χ⟩`` for functional [`J_T_sm`](@ref).

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
        w[k] = obj.weight
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
F_{\text{re}}
    = \Re[f_{τ}]
    = \Re\left[
        \frac{1}{N} \sum_{k=1}^{N} w_k τ_k
    \right]
    \quad\in \begin{cases}
    [-1, 1] & \text{in Hilbert space} \\
    [0, 1] & \text{in Liouville space.}
\end{cases}
```

with ``w_k`` the weight for the k'th objective and ``τ_k`` the overlap of the
k'th propagated state with the k'th target state, and ``N`` the number of
objectives.

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


@doc raw"""Backward boundary states ``|χ⟩`` for functional [`J_T_re`](@ref).

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
        w = obj.weight
        lmul!(w / (2N), χ[k])
    end
end




"""Convert a functional from acting on a gate to acting on propagated states.

```
J_T = gate_functional(J_T_U; kwargs...)
```

constructs a functional `J_T` that meets the requirements for
for Krotov/GRAPE and [`make_chi`](@ref). That is, the output `J_T` takes
positional positional arguments `ϕ` and `objectives`. The input functional
`J_T_U` is assumed to have the signature `J_T_U(U; kwargs...)` where `U` is a
matrix with elements ``U_{ij} = ⟨Ψ_i|ϕ_j⟩``, where ``|Ψ_i⟩`` is the
`initial_state` of the i'th `objectives` (assumed to be the i'th canonical
basis state) and ``|ϕ_j⟩`` is the result of forward-propagating ``|Ψ_j⟩``. That
is, `U` is the projection of the time evolution operator into the subspace
defined by the basis in the `initial_states` of the  `objectives`.

# See also

* [`make_gate_chi`](@ref) — create a corresponding `chi` function that acts
  more efficiently than the general [`make_chi`](@ref).
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


# default for `via` argument of `make_chi`
function _default_chi_via(objectives)
    if any(isnothing(obj.target_state) for obj in objectives)
        return :phi
    else
        return :tau
    end
end


@doc raw"""Return a function that evaluates ``|χ_k⟩ = -∂J_T/∂⟨ϕ_k|``.

```julia
chi! = make_chi(
    J_T,
    objectives;
    force_zygote=false,
    via=(any(isnothing(obj.target_state) for obj in objectives) ? :phi : :tau),
    use_finite_differences=false
)
```

creates a function `chi!(χ, ϕ, objectives; τ)` that sets
the k'th element of `χ` to ``|χ_k⟩ = -∂J_T/∂⟨ϕ_k|``, where ``|ϕ_k⟩`` is the
k'th element of `ϕ`. These are the states used as the boundary condition for
the backward propagation propagation in Krotov's method and GRAPE. Each
``|χₖ⟩`` is defined as a matrix calculus
[Wirtinger derivative](https://www.ekinakyurek.me/complex-derivatives-wirtinger/),

```math
|χ_k(T)⟩ = -\frac{∂J_T}{∂⟨ϕ_k|} = -\frac{1}{2} ∇_{ϕ_k} J_T\,;\qquad
∇_{ϕ_k} J_T ≡ \frac{∂J_T}{\Re[ϕ_k]} + i \frac{∂J_T}{\Im[ϕ_k]}\,.
```

The function `J_T` must take a vector of states `ϕ` and a vector of
`objectives` as positional parameters, and a vector `τ` as a keyword argument,
see e.g. [`J_T_sm`](@ref). If all objectives define a `target_state`, then `τ`
will be the overlap of the states `ϕ` with those target states. The functional
`J_T` may or may not use those overlaps.  Likewise, the resulting `chi!` may or
may not use the keyword parameter `τ`.

For functionals where ``-∂J_T/∂⟨ϕ_k|`` is known analytically, that analytic
derivative will be returned, e.g.,

* [`J_T_sm`](@ref) → [`chi_sm!`](@ref),
* [`J_T_re`](@ref) → [`chi_re!`](@ref),
* [`J_T_ss`](@ref) → [`chi_ss!`](@ref).

Otherwise, or if `force_zygote=true` or `use_finite_differences=true`, the
derivative to calculate ``|χ_k⟩`` will be evaluated automatically, via
automatic differentiation with Zygote, or via finite differences (which
primarily serves for testing the Zygote gradient).

When evaluating ``|χ_k⟩`` automatically, if `via=:phi` is given , ``|χ_k(T)⟩``
is calculated directly as defined a above from the gradient with respect to
the states ``\{|ϕ_k(T)⟩\}``. The resulting function `chi!` ignores any passed
`τ` keyword argument.

If `via=:tau` is given instead, the functional ``J_T`` is considered a function
of overlaps ``τ_k = ⟨ϕ_k^\tgt|ϕ_k(T)⟩``. This requires that all `objectives`
define a `target_state` and that `J_T` calculates the value of the functional
solely based on the values of `τ` passed as a keyword argument.  With only the
complex conjugate ``τ̄_k = ⟨ϕ_k(T)|ϕ_k^\tgt⟩`` having an explicit dependency on
``⟨ϕ_k(T)|``,  the chain rule in this case is

```math
|χ_k(T)⟩
= -\frac{∂J_T}{∂⟨ϕ_k|}
= -\left(
    \frac{∂J_T}{∂τ̄_k}
    \frac{∂τ̄_k}{∂⟨ϕ_k|}
  \right)
= - \frac{1}{2} (∇_{τ_k} J_T) |ϕ_k^\tgt⟩\,.
```

Again, we have used the definition of the Wirtinger derivatives,

```math
\begin{align*}
    \frac{∂J_T}{∂τ_k}
    &≡ \frac{1}{2}\left(
        \frac{∂ J_T}{∂ \Re[τ_k]}
        - i \frac{∂ J_T}{∂ \Im[τ_k]}
    \right)\,,\\
    \frac{∂J_T}{∂τ̄_k}
    &≡ \frac{1}{2}\left(
        \frac{∂ J_T}{∂ \Re[τ_k]}
        + i \frac{∂ J_T}{∂ \Im[τ_k]}
    \right)\,,
\end{align*}
```

and the definition of the Zygote gradient with respect to a complex scalar,

```math
∇_{τ_k} J_T = \left(
    \frac{∂ J_T}{∂ \Re[τ_k]}
    + i \frac{∂ J_T}{∂ \Im[τ_k]}
\right)\,.
```

!!! tip

    In order to extend `make_chi` with an analytic implementation for a new
    `J_T` function, define a new method `make_analytic_chi` like so:

    ```julia
    make_analytic_chi(::typeof(J_T_sm), objectives) = chi_sm!
    ```

    which links `make_chi` for [`J_T_sm`](@ref) to [`chi_sm!`](@ref).


!!! warning

    Zygote is notorious for being buggy (silently returning incorrect
    gradients). Always test automatic derivatives against finite differences
    and/or other automatic differentiation frameworks.
"""
function make_chi(
    J_T,
    objectives;
    force_zygote=false,
    via=_default_chi_via(objectives),
    use_finite_differences=false
)
    if (force_zygote || use_finite_differences)
        return make_automatic_chi(J_T, objectives; via, use_finite_differences)
    else
        return make_analytic_chi(J_T, objectives)
    end
end


function make_automatic_chi(
    J_T,
    objectives;
    via=_default_chi_via(objectives),
    use_finite_differences=false
)

    N = length(objectives)

    # TODO: keyword arguments (τ) should not use unicode

    function zygote_chi_via_phi!(χ, ϕ, objectives; τ=nothing)
        function _J_T(Ψ...)
            -J_T(Ψ, objectives)
        end
        if use_finite_differences
            fdm = FiniteDifferences.central_fdm(5, 1)
            ∇J = FiniteDifferences.grad(fdm, _J_T, ϕ...)
        else
            ∇J = Zygote.gradient(_J_T, ϕ...)
        end
        for (k, ∇Jₖ) ∈ enumerate(∇J)
            # |χₖ⟩ = ½ |∇Jₖ⟩  # ½ corrects for gradient vs Wirtinger deriv
            axpby!(0.5, ∇Jₖ, false, χ[k])
        end
    end

    function zygote_chi_via_tau!(χ, ϕ, objectives; τ)
        function _J_T(τ...)
            -J_T(ϕ, objectives; τ=τ)
        end
        if use_finite_differences
            fdm = FiniteDifferences.central_fdm(5, 1)
            ∇J = FiniteDifferences.grad(fdm, _J_T, τ...)
        else
            ∇J = Zygote.gradient(_J_T, τ...)
        end
        for (k, ∇Jₖ) ∈ enumerate(∇J)
            ∂J╱∂τ̄ₖ = 0.5 * ∇Jₖ  # ½ corrects for gradient vs Wirtinger deriv
            # |χₖ⟩ = (∂J/∂τ̄ₖ) |ϕₖ⟩
            axpby!(∂J╱∂τ̄ₖ, objectives[k].target_state, false, χ[k])
        end
    end

    # Test J_T function interface
    ϕ_initial = [obj.initial_state for obj in objectives]
    J_T_val = J_T(ϕ_initial, objectives)
    J_T_val::Float64

    if via ≡ :phi
        return zygote_chi_via_phi!
    elseif via ≡ :tau
        ϕ_tgt = [obj.target_state for obj in objectives]
        if any(isnothing.(ϕ_tgt))
            error("`via=:tau` requires that all objectives define a `target_state`")
        end
        τ_tgt = ComplexF64[1.0 for obj in objectives]
        if abs(J_T(ϕ_tgt, objectives) - J_T(nothing, objectives; τ=τ_tgt)) > 1e-12
            error(
                "`via=:tau` in `make_chi` requires that `J_T`=$(repr(J_T)) can be evaluated solely via `τ`"
            )
        end
        return zygote_chi_via_tau!
    else
        error("`via` must be either `:phi` or `:tau`, not $(repr(via))")
    end

end

make_analytic_chi(J_T, objectives) = make_automatic_chi(J_T, objectives)
# Well, the above fallback to `make_automatic_chi` is clearly not "analytic", but
# the intent of this name is to allow users to define new analytic
# implementation, cf. the explanation in the doc of `make_chi`.
make_analytic_chi(::typeof(J_T_sm), objectives) = chi_sm!
make_analytic_chi(::typeof(J_T_re), objectives) = chi_re!
make_analytic_chi(::typeof(J_T_ss), objectives) = chi_ss!


end
