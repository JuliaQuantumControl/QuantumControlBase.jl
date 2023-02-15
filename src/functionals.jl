using LinearAlgebra
using Zygote: Zygote
using FiniteDifferences: FiniteDifferences


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
see e.g. `J_T_sm`). If all objectives define a `target_state`, then `τ`
will be the overlap of the states `ϕ` with those target states. The functional
`J_T` may or may not use those overlaps.  Likewise, the resulting `chi!` may or
may not use the keyword parameter `τ`.

For functionals where ``-∂J_T/∂⟨ϕ_k|`` is known analytically, that analytic
derivative will be returned, e.g.,

* `J_T_sm` → `chi_sm!`,
* `J_T_re` → `chi_re!`,
* `J_T_ss` → `chi_ss!`.

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

    which links `make_chi` for `J_T_sm` to `chi_sm!`.


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
            ∇Jₖ = convert(typeof(χ[k]), ∇Jₖ)
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


"""
Return a function to evaluate ``∂J_a/∂ϵ_{ln}`` for a pulse value running cost.

```julia
grad_J_a! = make_grad_J_a(
    J_a,
    tlist;
    force_zygote=false,
    use_finite_differences=false
)
```

returns a function so that `grad_J_a!(∇J_a, pulsevals, tlist)` sets
``∂J_a/∂ϵ_{ln}`` as the elements of the (vectorized) `∇J_a`. The function `J_a`
must have the interface `J_a(pulsevals, tlist)`, see, e.g.,
`J_a_fluence`.

If `force_zygote=true`, automatic differentiation with Zygote will be used to
calculate the derivative.

If `use_finite_differences=true`, the derivative will be calculated via finite
differences. This may be used to verify Zygote gradients.

By default, for functionals `J_a` that have a known analytic derivative, that
analytic derivative will be used. For unknown functions, the derivative will be
calculated via Zygote.

!!! tip

    In order to extend `make_grad_J_a` with an analytic implementation for a
    new `J_a` function, define a new method `make_analytic_grad_J_a` like so:

    ```julia
    make_analytic_grad_J_a(::typeof(J_a_fluence), tlist) = grad_J_a_fluence!
    ```

    which links `make_grad_J_a` for `J_a_fluence` to `grad_J_a_fluence!`.
"""
function make_grad_J_a(J_a, tlist; force_zygote=false, use_finite_differences=false)
    if (force_zygote || use_finite_differences)
        return make_automatic_grad_J_a(J_a, tlist; use_finite_differences)
    else
        return make_analytic_grad_J_a(J_a, tlist)
    end
end


function make_automatic_grad_J_a(J_a, tlist; use_finite_differences=false)
    function automatic_grad_J_a!(∇J_a, pulsevals, tlist)
        func = pulsevals -> J_a(pulsevals, tlist)
        if use_finite_differences
            fdm = FiniteDifferences.central_fdm(5, 1)
            ∇J_a_fdm = FiniteDifferences.grad(fdm, func, pulsevals)[1]
            copyto!(∇J_a, ∇J_a_fdm)
        else
            ∇J_a_zygote = Zygote.gradient(func, pulsevals)[1]
            copyto!(∇J_a, ∇J_a_zygote)
        end
    end
    return automatic_grad_J_a!
end

make_analytic_grad_J_a(J_a, tlist) = make_automatic_grad_J_a(J_a, tlist)
