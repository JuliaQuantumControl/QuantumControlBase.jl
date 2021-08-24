"""Extension of QuantumPropagators.propagate for control objectives."""

import QuantumPropagators


"""Propagate the initial state of a control objective.

```julia
propagate(obj, tlist; controls_map=IdDict(), kwargs...)
```

propagates `obj.initial_state` under the dynamics described by `obj.generator`.

The optional dict `control_map` may be given to replace the controls in
`obj.generator` (as obtained by [`getcontrols`](@ref)) with custom functions or
vectors, e.g. with the controls resulting from optimization.

All `kwargs` are forwarded to `QuantumPropagators.propagate`.
"""
function QuantumPropagators.propagate(
    obj::Objective, tlist; controls_map=IdDict(), kwargs...
)

    controls = getcontrols(obj.generator)
    pulses = [
        discretize_on_midpoints(get(controls_map, control, control), tlist)
        for control in controls
    ]

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
