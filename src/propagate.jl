"""Extension of QuantumPropagators.propagate for control objectives."""

import QuantumPropagators


"""Propagate the initial state of a control objective.

```julia
propagate(obj, tlist, method=:auto; controls_map=IdDict(), kwargs...)
```

propagates `obj.initial_state` under the dynamics described by `obj.generator`.

The optional dict `control_map` may be given to replace the controls in
`obj.generator` (as obtained by [`getcontrols`](@ref)) with custom functions or
vectors, e.g. with the controls resulting from optimization.

All `kwargs` are forwarded to `QuantumPropagators.propagate`.
"""
function QuantumPropagators.propagate(
    obj::Objective, tlist, method=Val(:auto); controls_map=IdDict(), kwargs...
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

    wrk = QuantumPropagators.initpropwrk(obj, tlist; method=method, kwargs...)
    return QuantumPropagators.propagate(obj.initial_state, genfunc, tlist, wrk;
                                        kwargs...)

end


"""
```julia
wrk = initpropwrk(obj, tlist; method=:auto, kwargs...)
```

initializes a workspace for the propagation of a control [`Objective`](@ref).

Note that `method` must be given as a keyword argument.
"""
function QuantumPropagators.initpropwrk(obj::Objective, tlist;
                                        method=Val(:auto), kwargs...)
    # method is a kw-arg because otherwise this method is ambiguous with
    # initpropwrk(state, tlist, method::Val{:auto}, generator...; kwargs...)
    state = obj.initial_state
    controls = getcontrols(obj.generator)
    controlvals = [discretize(control, tlist) for control in controls]
    zero_vals = IdDict(
        control => zero(controlvals[i][1])
        for (i, control) in enumerate(controls)
    )
    G_zero = setcontrolvals(obj.generator, zero_vals)
    max_vals = IdDict(
        control => maximum(controlvals[i])
        for (i, control) in enumerate(controls)
    )
    G_max = setcontrolvals(obj.generator, max_vals)
    min_vals = IdDict(
        control => minimum(controlvals[i])
        for (i, control) in enumerate(controls)
    )
    G_min = setcontrolvals(obj.generator, min_vals)
    return QuantumPropagators.initpropwrk(
        state, tlist, method, G_zero, G_max, G_min; kwargs...
    )
end
