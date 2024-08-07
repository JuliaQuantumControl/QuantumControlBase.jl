using LinearAlgebra


# default for `via` argument of `make_chi`
function _default_chi_via(trajectories)
    if any(isnothing(traj.target_state) for traj in trajectories)
        return :states
    else
        return :tau
    end
end


"""Overlaps of target states with propagates states

```julia
τ = taus(Ψ, trajectories)
```

calculates a vector of values ``τ_k = ⟨Ψ_k^{tgt}|Ψ_k⟩`` where ``|Ψ_k^{tgt}⟩``
is the `traj.target_state` of the ``k``'th element of `trajectories` and
``|Ψₖ⟩`` is the ``k``'th element of `Ψ`.

The definition of the τ values with ``Ψ_k^{tgt}`` on the left (overlap of
target states with propagated states, as opposed to overlap of propagated
states with target states) matches Refs. [PalaoPRA2003](@cite)
and [GoerzQ2022](@cite).

The function requires that each trajectory defines a target state.
See also [`taus!`](@ref) for an in-place version that includes well-defined
error handling for any trajectories whose `target_state` property is `nothing`.
"""
function taus(Ψ, trajectories)
    # This function does not delegate to `taus!`, in order to make it
    # compatible with automatic differentiation, which doesn't support
    # mutation.
    return [dot(traj.target_state, Ψₖ) for (traj, Ψₖ) in zip(trajectories, Ψ)]
end


"""Overlaps of target states with propagates states, calculated in-place.

```julia
taus!(τ, Ψ, trajectories; ignore_missing_target_state=false)
```

overwrites the complex vector `τ` with the results of
[`taus(Ψ, trajectories)`](@ref taus).

Throws an `ArgumentError` if any of trajectories have a `target_state` of
`nothing`. If `ignore_missing_target_state=true`, values in `τ` instead will
remain unchanged for any trajectories with a missing target state.
"""
function taus!(τ::Vector{ComplexF64}, Ψ, trajectories; ignore_missing_target_state=false)
    for (k, (traj, Ψₖ)) in enumerate(zip(trajectories, Ψ))
        if !isnothing(traj.target_state)
            τ[k] = dot(traj.target_state, Ψₖ)
        else
            # With `ignore_missing_target_state=true`, we just skip the value.
            # This makes `taus!` convenient for calculating τ values in
            # Krotov/GRAPE if and only if the function is based on target
            # states
            if !ignore_missing_target_state
                msg = "trajectory[$k] has no `target_state`. Cannot calculate τ = ⟨Ψ_tgt|Ψ⟩"
                throw(ArgumentError(msg))
            end
        end
    end

end


@doc raw"""Return a function that calculates ``|χ_k⟩ = -∂J_T/∂⟨Ψ_k|``.

```julia
chi = make_chi(
    J_T,
    trajectories;
    mode=:any,
    automatic=:default,
    via=(any(isnothing(t.target_state) for t in trajectories) ? :states : :tau),
)
```

creates a function `chi(Ψ, trajectories; τ)` that returns
a vector of states `χ` with ``|χ_k⟩ = -∂J_T/∂⟨Ψ_k|``, where ``|Ψ_k⟩`` is the
k'th element of `Ψ`. These are the states used as the boundary condition for
the backward propagation propagation in Krotov's method and GRAPE. Each
``|χₖ⟩`` is defined as a matrix calculus
[Wirtinger derivative](https://www.ekinakyurek.me/complex-derivatives-wirtinger/),

```math
|χ_k(T)⟩ = -\frac{∂J_T}{∂⟨Ψ_k|} = -\frac{1}{2} ∇_{Ψ_k} J_T\,;\qquad
∇_{Ψ_k} J_T ≡ \frac{∂J_T}{\Re[Ψ_k]} + i \frac{∂J_T}{\Im[Ψ_k]}\,.
```

The function `J_T` must take a vector of states `Ψ` and a vector of
`trajectories` as positional parameters. If `via=:tau`, it must also a vector
`tau` as a keyword argument, see e.g. `J_T_sm`).
that contains the overlap of the states `Ψ` with the target states from the `trajectories`

The derivative can be calculated analytically of automatically (via automatic
differentiation) depending on the value of `mode`. For `mode=:any`, an analytic
derivative is returned if available, with a fallback to an automatic derivative.

If `mode=:analytic`, return an analytically known ``-∂J_T/∂⟨Ψ_k|``, e.g.,

* [`QuantumControl.Functionals.J_T_sm`](@ref) → [`QuantumControl.Functionals.chi_sm`](@ref),
* [`QuantumControl.Functionals.J_T_re`](@ref) → [`QuantumControl.Functionals.chi_re`](@ref),
* [`QuantumControl.Functionals.J_T_ss`](@ref) → [`QuantumControl.Functionals.chi_ss`](@ref).

and throw an error if no analytic derivative is known.

If `mode=:automatic`, return an automatic derivative (even if an analytic
derivative is known). The calculation of an automatic derivative  (whether via
`mode=:any` or `mode=:automatic`) requires that a suitable framework (e.g.,
`Zygote` or `FiniteDifferences`) has been loaded. The loaded module must be
passed as `automatic` keyword argument. Alternatively, it can be registered as
a default value for `automatic` by calling
`QuantumControl.set_default_ad_framework`.

When evaluating ``|χ_k⟩`` automatically, if `via=:states` is given , ``|χ_k(T)⟩``
is calculated directly as defined above from the gradient with respect to
the states ``\{|Ψ_k(T)⟩\}``.

If `via=:tau` is given instead, the functional ``J_T`` is considered a function
of overlaps ``τ_k = ⟨Ψ_k^\tgt|Ψ_k(T)⟩``. This requires that all `trajectories`
define a `target_state` and that `J_T` calculates the value of the functional
solely based on the values of `tau` passed as a keyword argument.  With only
the complex conjugate ``τ̄_k = ⟨Ψ_k(T)|Ψ_k^\tgt⟩`` having an explicit dependency
on ``⟨Ψ_k(T)|``,  the chain rule in this case is

```math
|χ_k(T)⟩
= -\frac{∂J_T}{∂⟨Ψ_k|}
= -\left(
    \frac{∂J_T}{∂τ̄_k}
    \frac{∂τ̄_k}{∂⟨Ψ_k|}
  \right)
= - \frac{1}{2} (∇_{τ_k} J_T) |Ψ_k^\tgt⟩\,.
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
    QuantumControlBase.make_analytic_chi(::typeof(J_T_sm), trajectories) = chi_sm
    ```

    which links `make_chi` for [`QuantumControl.Functionals.J_T_sm`](@ref)
    to [`QuantumControl.Functionals.chi_sm`](@ref).


!!! warning

    Zygote is notorious for being buggy (silently returning incorrect
    gradients). Always test automatic derivatives against finite differences
    and/or other automatic differentiation frameworks.
"""
function make_chi(
    J_T,
    trajectories;
    mode=:any,
    automatic=:default,
    via=_default_chi_via(trajectories),
)
    if mode == :any
        try
            chi = make_analytic_chi(J_T, trajectories)
            @debug "make_chi for J_T=$(J_T) -> analytic"
            # TODO: call chi to compile it and ensure required properties
            return chi
        catch exception
            if exception isa MethodError
                @info "make_chi for J_T=$(J_T): fallback to mode=:automatic"
                try
                    chi = make_automatic_chi(J_T, trajectories, automatic; via)
                    # TODO: call chi to compile it and ensure required properties
                    return chi
                catch exception
                    if exception isa MethodError
                        msg = "make_chi for J_T=$(J_T): no analytic gradient, and no automatic gradient with `automatic=$(repr(automatic))`."
                        error(msg)
                    else
                        rethrow()
                    end
                end
            else
                rethrow()
            end
        end
    elseif mode == :analytic
        try
            chi = make_analytic_chi(J_T, trajectories)
            # TODO: call chi to compile it and ensure required properties
            return chi
        catch exception
            if exception isa MethodError
                msg = "make_chi for J_T=$(J_T): no analytic gradient. Implement `QuantumControlBase.make_analytic_chi(::typeof(J_T), trajectories)`"
                error(msg)
            else
                rethrow()
            end
        end
    elseif mode == :automatic
        try
            chi = make_automatic_chi(J_T, trajectories, automatic; via)
            # TODO: call chi to compile it and ensure required properties
            return chi
        catch exception
            if exception isa MethodError
                msg = "make_chi for J_T=$(J_T): no automatic gradient with `automatic=$(repr(automatic))`."
                error(msg)
            else
                rethrow()
            end
        end
    else
        msg = "`mode=$(repr(mode))` must be one of :any, :analytic, :automatic"
        throw(ArgumentError(msg))
    end
end


# Generic placeholder
function make_analytic_chi end


# Module to Symbol-Val
function make_automatic_chi(J_T, trajectories, automatic::Module; via)
    return make_automatic_chi(J_T, trajectories, Val(nameof(automatic)); via)
end

# Symbol to Symbol-Val
function make_automatic_chi(J_T, trajectories, automatic::Symbol; via)
    return make_automatic_chi(J_T, trajectories, Val(automatic); via)
end


DEFAULT_AD_FRAMEWORK = :nothing

function make_automatic_chi(J_T, trajectories, ::Val{:default}; via)
    if DEFAULT_AD_FRAMEWORK == :nothing
        msg = "make_chi: no default `automatic`. You must run `QuantumControl.set_default_ad_framework` first, e.g. `import Zygote; QuantumControl.set_default_ad_framework(Zygote)`."
        error(msg)
    else
        automatic = DEFAULT_AD_FRAMEWORK
        chi = make_automatic_chi(J_T, trajectories, DEFAULT_AD_FRAMEWORK; via)
        (string(automatic) == "default") && error("automatic fallback")
        @info "make_chi for J_T=$(J_T): automatic with $automatic"
        return chi
    end
end


# There is a user-facing wrapper `QuantumControl.set_default_ad_framework`.
# See the documentation there.
function _set_default_ad_framework(mod::Module; quiet=false)
    global DEFAULT_AD_FRAMEWORK
    automatic = nameof(mod)
    if !quiet
        @info "QuantumControlBase: Setting $automatic as the default provider for automatic differentiation."
    end
    DEFAULT_AD_FRAMEWORK = automatic
    return nothing
end


function _set_default_ad_framework(::Nothing; quiet=false)
    global DEFAULT_AD_FRAMEWORK
    if !quiet
        @info "Unsetting the default provider for automatic differentiation."
    end
    DEFAULT_AD_FRAMEWORK = :nothing
    return nothing
end



"""
Return a function to evaluate ``∂J_a/∂ϵ_{ln}`` for a pulse value running cost.

```julia
grad_J_a! = make_grad_J_a(
    J_a,
    tlist;
    mode=:any,
    automatic=:default,
)
```

returns a function so that `grad_J_a!(∇J_a, pulsevals, tlist)` sets
``∂J_a/∂ϵ_{ln}`` as the elements of the (vectorized) `∇J_a`. The function `J_a`
must have the interface `J_a(pulsevals, tlist)`, see, e.g.,
`J_a_fluence`.

The parameters `mode` and `automatic` are handled as in [`make_chi`](@ref),
where `mode` is one of `:any`, `:analytic`, `:automatic`, and `automatic` is
he loaded module of an automatic differentiation framework, where `:default`
refers to the framework set with `QuantumControl.set_default_ad_framework`.

!!! tip

    In order to extend `make_grad_J_a` with an analytic implementation for a
    new `J_a` function, define a new method `make_analytic_grad_J_a` like so:

    ```julia
    make_analytic_grad_J_a(::typeof(J_a_fluence), tlist) = grad_J_a_fluence!
    ```

    which links `make_grad_J_a` for `J_a_fluence` to `grad_J_a_fluence!`.
"""
function make_grad_J_a(J_a, tlist; mode=:any, automatic=:default)
    if mode == :any
        try
            grad_J_a = make_analytic_grad_J_a(J_a, tlist)
            @debug "make_grad_J_a for J_a=$(J_a) -> analytic"
            return grad_J_a
        catch exception
            if exception isa MethodError
                @info "make_grad_J_a for J_a=$(J_a): fallback to mode=:automatic"
                try
                    grad_J_a = make_automatic_grad_J_a(J_a, tlist, automatic)
                    return grad_J_a
                catch exception
                    if exception isa MethodError
                        msg = "make_grad_J_a for J_a=$(J_a): no analytic gradient, and no automatic gradient with `automatic=$(repr(automatic))`."
                        error(msg)
                    else
                        rethrow()
                    end
                end
            else
                rethrow()
            end
        end
    elseif mode == :analytic
        try
            return make_analytic_grad_J_a(J_a, tlist)
        catch exception
            if exception isa MethodError
                msg = "make_grad_J_a for J_a=$(J_a): no analytic gradient. Implement `QuantumControlBase.make_analytic_grad_J_a(::typeof(J_a), tlist)`"
                error(msg)
            else
                rethrow()
            end
        end
    elseif mode == :automatic
        try
            return make_automatic_grad_J_a(J_a, tlist, automatic)
        catch exception
            if exception isa MethodError
                msg = "make_grad_J_a for J_a=$(J_a): no automatic gradient with `automatic=$(repr(automatic))`."
                error(msg)
            else
                rethrow()
            end
        end
    else
        msg = "`mode=$(repr(mode))` must be one of :any, :analytic, :automatic"
        throw(ArgumentError(msg))
    end
end


function make_automatic_grad_J_a(J_a, tlist, ::Val{:default})
    if DEFAULT_AD_FRAMEWORK == :nothing
        msg = "make_automatic_grad_J_a: no default `automatic`. You must run `set_default_ad_framework` first, e.g. `import Zygote; QuantumControl.set_default_ad_framework(Zygote)`."
        error(msg)
    else
        automatic = DEFAULT_AD_FRAMEWORK
        grad_J_a = make_automatic_grad_J_a(J_a, tlist, DEFAULT_AD_FRAMEWORK)
        @info "make_grad_J_a for J_a=$(J_a): automatic with $automatic"
        return grad_J_a
    end
end

# Generic placeholder
function make_analytic_grad_J_a end

# Module to Symbol-Val
function make_automatic_grad_J_a(J_a, tlist, automatic::Module)
    return make_automatic_grad_J_a(J_a, tlist, Val(nameof(automatic)))
end

# Symbol to Symbol-Val
function make_automatic_grad_J_a(J_a, tlist, automatic::Symbol)
    return make_automatic_grad_J_a(J_a, tlist, Val(automatic))
end
