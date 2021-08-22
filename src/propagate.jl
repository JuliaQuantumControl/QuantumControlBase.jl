"""Extension of QuantumPropagators.propagate for control objectives."""

import QuantumPropagators


"""Propagate the initial state of a control objective.

```julia
propagate(obj; kwargs...)
```

All keyword arguments are forwarded to `QuantumPropagators.propagate`.
"""
function QuantumPropagators.propagate(obj::Objective, tlist; kwargs...)

    controls = getcontrols(obj.generator)
    pulses = [discretize_on_midpoints(control, tlist) for control in controls]

    zero_vals = IdDict(control => 0 for control in controls)
    G = setcontrolvals(obj.generator, zero_vals)

    function genfunc(tlist, i; kwargs...)
        vals_dict = IdDict(
            control => pulses[j][i] for (j, control) in enumerate(controls)
        )
        setcontrolvals!(G, obj.generator, vals_dict)
        return G
    end

    return QuantumPropagators.propagate(obj.initial_state, genfunc, tlist;
                                        kwargs...)

end
