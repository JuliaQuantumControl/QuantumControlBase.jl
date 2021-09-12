"""Extension of QuantumPropagators.propagate for control objectives."""

import QuantumPropagators


"""Propagate the initial state of a control objective.

```julia
propagate(obj, tlist; method=:auto, controls_map=IdDict(), kwargs...)
```

propagates `obj.initial_state` under the dynamics described by `obj.generator`.

The optional dict `control_map` may be given to replace the controls in
`obj.generator` (as obtained by [`getcontrols`](@ref)) with custom functions or
vectors, e.g. with the controls resulting from optimization.

All other `kwargs` are forwarded to the underlying `propagate` method for
`obj.initial_state`.
"""
function QuantumPropagators.propagate(
    obj::OT, tlist; method=Val(:auto), controls_map=IdDict(), kwargs...
) where {OT <: AbstractControlObjective}
    return propagate_objective(obj, tlist, method;
                               controls_map=controls_map, kwargs...)
end


function propagate_objective(
    obj::OT, tlist, method::Symbol; controls_map=IdDict(), kwargs...
) where {OT <: AbstractControlObjective}
    return propagate_objective(obj, tlist, Val(method);
                               controls_map=controls_map, kwargs...)
end


function propagate_objective(
    obj::OT, tlist, method::Val; controls_map=IdDict(), kwargs...
) where {OT <: AbstractControlObjective}
    wrk = initobjpropwrk(obj, tlist, method; kwargs...)
    return propagate_objective_with_wrk(obj, tlist, wrk; kwargs...)

end


function propagate_objective_with_wrk(
    obj::OT, tlist, wrk; controls_map=IdDict(), kwargs...
) where {OT <: AbstractControlObjective}

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

    return QuantumPropagators.propagate_state_with_wrk(
            obj.initial_state, genfunc, tlist, wrk; kwargs...
    )

end


"""
```julia
wrk = initobjpropwrk(obj, tlist, method; kwargs...)
```

initializes a workspace for the propagation of
an[`AbstractControlObjective`](@ref).
"""
function initobjpropwrk(
            obj::OT, tlist, method::Val; kwargs...
        ) where {OT <: AbstractControlObjective}
    # This doesn't extend QuantumPropagators.initpropwrk directly, because it
    # would lead to an "ambiguous method"
    state = obj.initial_state
    controls = getcontrols(obj.generator)
    controlvals = [discretize(control, tlist) for control in controls]
    # We'll construct some examplary dynamical generators. This is mainly for
    # method=:cheby, but it's the "safe" thing to do for any method that uses
    # generatores for initpropwrk, so we're using this as the default
    # implementation
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


function initobjpropwrk(
            obj::OT, tlist, method::Symbol; kwargs...
        ) where {OT <: AbstractControlObjective}
    return  initobjpropwrk(obj::OT, tlist, Val(method); kwargs...)
end


function initobjpropwrk(
            obj::OT, tlist, method::Val{:newton}; kwargs...
        ) where {OT <: AbstractControlObjective}
    state = obj.initial_state
    return QuantumPropagators.initpropwrk(
        state, tlist, method; kwargs...
    )
end


function initobjpropwrk(
            obj::OT, tlist, method::Val{:expprop}; kwargs...
        ) where {OT <: AbstractControlObjective}
    state = obj.initial_state
    return QuantumPropagators.initpropwrk(
        state, tlist, method; kwargs...
    )
end
