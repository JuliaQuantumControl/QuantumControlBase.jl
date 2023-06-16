using QuantumPropagators
using QuantumPropagators.Generators: Generator
using QuantumPropagators.Controls: get_controls
# from ./derivs.jl:  get_control_derivs, get_control_deriv


"""Check the dynamical `generator` in the context of optimal control.

```
@test check_generator(generator; state, tlist,
                     for_mutable_state=true, for_immutable_state=true,
                     for_expval=true, for_gradient_optimization=true,
                     atol=1e-15)
```

verifies the given `generator`. This checks all the conditions of
[`QuantumPropagators.Interfaces.check_generator`](@ref). In addition, the
following conditions must be met.

If `for_gradient_optimization`:

* [`get_control_derivs(generator, controls)`](@ref get_control_derivs) must be
  defined and return a vector containing the result of
  [`get_control_deriv(generator, control)`](@ref get_control_deriv) for every
  `control` in `controls`.
* [`get_control_deriv(generator, control)`](@ref get_control_deriv) must return
  an object that passes the less restrictive
  [`QuantumPropagators.Interfaces.check_generator`](@ref) if `control` is in
  `get_controls(generator)`.
* [`get_control_deriv(generator, control)`](@ref get_control_deriv) must return
  `nothing` if `control` is not in
  [`get_controls(generator)`](@ref get_controls)
* If `generator` is a [`Generator`](@ref) instance, every `ampl` in
  `generator.amplitudes` must pass [`check_amplitude(ampl; tlist)`](@ref
  check_amplitude).
"""
function check_generator(
    generator;
    state,
    tlist,
    for_mutable_state=true,
    for_immutable_state=true,
    for_expval=true,
    for_gradient_optimization=true,
    atol=1e-15
)

    success = QuantumPropagators.Interfaces.check_generator(
        generator;
        state,
        tlist,
        for_mutable_state,
        for_immutable_state,
        for_expval,
        atol,
        _check_amplitudes=false  # amplitudes are checked separately
    )
    success || (return false)

    if for_gradient_optimization

        try
            controls = get_controls(generator)
            control_derivs = get_control_derivs(generator, controls)
            if !(control_derivs isa Vector)
                @error "`get_control_derivs(generator, controls)` must return a Vector"
                success = false
            end
            if length(control_derivs) ≠ length(controls)
                @error "`get_control_derivs(generator, controls)` must return a derivative for every `control` in `controls`"
                success = false
            end
            # In general, we can't check for equality between
            # `get_control_deriv` and `get_control_deriv`, because `==` may not
            # be implemented to compare arbitrary generators by value
        catch exc
            @error "`get_control_derivs(generator, controls)` must be defined: $exc"
            success = false
        end

        try
            controls = get_controls(generator)
            for (i, control) in enumerate(controls)
                deriv = get_control_deriv(generator, control)
                valid_deriv = check_generator(
                    deriv;
                    state,
                    tlist,
                    for_mutable_state,
                    for_immutable_state,
                    for_expval,
                    atol,
                    for_gradient_optimization=false
                )
                if !valid_deriv
                    @error "the result of `get_control_deriv(generator, control)` for control $i is not a valid generator"
                    success = false
                end
            end
        catch exc
            @error "`get_control_deriv(generator, control)` must be defined: $exc"
            success = false
        end

        try
            controls = get_controls(generator)
            dummy_control_CYRmE(t) = rand()
            @assert dummy_control_CYRmE ∉ controls
            deriv = get_control_deriv(generator, dummy_control_CYRmE)
            if deriv ≢ nothing
                @error "`get_control_deriv(generator, control)` must return `nothing` if `control` is not in `get_controls(generator)`, not $(repr(deriv))"
                success = false
            end
        catch exc
            @error "`get_control_deriv(generator, control)` must return `nothing` if `control` is not in `get_controls(generator)`: $exc"
            success = false
        end

        if generator isa Generator
            for (i, ampl) in enumerate(generator.amplitudes)
                if !check_amplitude(ampl; tlist, for_gradient_optimization)
                    success = false
                end
            end
        end

    end

    return success

end
