using QuantumPropagators
using QuantumPropagators.Controls: get_controls
# from ./derivs.jl:  get_control_deriv


"""
Check an amplitude in a [`Generator`](@ref) in the context of optimal control.

```
@test check_amplitude(
    ampl; tlist, for_gradient_optimization=true, quiet=false
)
```

verifies that the given `ampl` is a valid element in the list of `amplitudes`
of a [`Generator`](@ref) object. This checks all the conditions of
[`QuantumPropagators.Interfaces.check_amplitude`](@ref). In addition, the
following conditions must be met.

If `for_gradient_optimization`:

* The function [`get_control_deriv(ampl, control)`](@ref get_control_deriv)
  must be defined
* If `ampl` does not depend on `control`, [`get_control_deriv(ampl,
  control)`](@ref get_control_deriv) must return `0.0`
* If `ampl` depends on `control`,
  [`get_control_deriv(ampl, control)`](@ref get_control_deriv) must return an
  object `u` so that `evaluate(u, tlist, n)` returns a Number. In most cases,
  `u` itself will be a Number.

The function returns `true` for a valid amplitude and `false` for an invalid
amplitude. Unless `quiet=true`, it will log an error to indicate which of the
conditions failed.
"""
function check_amplitude(
    ampl;
    tlist,
    for_gradient_optimization=true,
    quiet=false,
    _message_prefix=""  # for recursive calling
)

    px = _message_prefix
    success =
        QuantumPropagators.Interfaces.check_amplitude(ampl; tlist, quiet, _message_prefix)
    success || (return false)

    if for_gradient_optimization

        controls = get_controls(ampl)  # guaranteed to work if success still true
        dummy_control_aSQeB(t) = rand()
        for (j, control) in enumerate(controls)
            try
                deriv = get_control_deriv(ampl, control)
                val = evaluate(deriv, tlist, 1)
                if !(val isa Number)
                    quiet ||
                        @error "$(px)get_control_deriv(ampl, control) for  control $j must return an object that evaluates to a Number, not $(typeof(val))"
                    success = false
                end
            catch exc
                quiet ||
                    @error "$(px)get_control_deriv(ampl, control) must be defined for control $j: $exc"
                success = false
            end
        end
        @assert dummy_control_aSQeB ∉ controls
        try
            deriv = get_control_deriv(ampl, dummy_control_aSQeB)
            if deriv ≠ 0.0
                quiet ||
                    @error "$(px)get_control_deriv(ampl, control) must return 0.0 if it does not depend on `control`, not $(repr(deriv))"
                success = false
            end
        catch exc
            quiet || @error "$(px)get_control_deriv(ampl, control) must be defined: $exc"
            success = false
        end

    end

    return success

end
