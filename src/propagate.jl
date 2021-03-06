"""Extension of QuantumPropagators.propagate for control objectives."""

import QuantumPropagators
import .ConditionalThreads: @threadsif

"""Construct a `genfunc` suitable for propagating an objective.

```julia
genfunc = objective_genfunc(obj, tlist; controls_map=IdDict())
```

can be passed to [`QuantumPropagators.propagate`](@ref) to propagate under the
dynamical generator in `obj`.

If given, `control_map` defines replacements for the controls in
`obj.generator`. This allows, e.g., to replace the controls with those
resulting from an optimization.
"""
function objective_genfunc(obj::AbstractControlObjective, tlist; controls_map=IdDict())
    controls = getcontrols(obj.generator)
    pulses = [
        discretize_on_midpoints(get(controls_map, control, control), tlist) for
        control in controls
    ]

    zero_vals = IdDict(control => 0 for control in controls)
    G = evalcontrols(obj.generator, zero_vals)
    vals_dict = IdDict(control => pulses[j][1] for (j, control) in enumerate(controls))

    function genfunc(tlist, i; kwargs...)
        for (j, control) in enumerate(controls)
            vals_dict[control] = pulses[j][i]
        end
        evalcontrols!(G, obj.generator, vals_dict)
        return G
    end

    return genfunc
end


# Internal method for getting a propagation method from one or more keyword
# arguments, with a default in a corresponding field/property of `obj`. For use
# in setting up the workspace for optimization methods.
#
# For example,
#
#     get_objective_prop_method(obj, :bw_prop_method, :prop_method; kwargs...)
#
# uses `kwargs[:bw_prop_method]`, `obj.bw_prop_method`, `kwargs[:prop_method]`,
# `obj.prop_method`, `:auto`, in that order.
#
function get_objective_prop_method(obj, symbols...; kwargs...)
    for symbol in symbols
        if symbol ∈ keys(kwargs)
            return kwargs[symbol]
        elseif symbol ∈ propertynames(obj)
            return Symbol(getproperty(obj, symbol))
        end
    end
    return :auto
end


"""Propagate with the dynamical generator of a control objective.

```julia
propagate_objective(obj, tlist; method=:auto, initial_state=obj.initial_state,
                    controls_map=IdDict(), kwargs...)
```

propagates `initial_state` under the dynamics described by `obj.generator`.

The optional dict `control_map` may be given to replace the controls in
`obj.generator` (as obtained by [`getcontrols`](@ref)) with custom functions or
vectors, e.g. with the controls resulting from optimization.

If `obj` has a property/field `prop_method` or `fw_prop_method`, its value will
be used as the default for `method` instead of :auto. An explicit keyword
argument for `method` always overrides the default.

All other `kwargs` are forwarded to the underlying
[`QuantumPropagators.propagate`](@ref) method for `obj.initial_state`.
"""
function propagate_objective(
    obj,
    tlist;
    initial_state=obj.initial_state,
    controls_map=IdDict(),
    kwargs...
)
    method = :auto
    if :method in keys(kwargs)
        method = kwargs[:method]
    else
        for symbol ∈ (:prop_method, :fw_prop_method)
            if symbol ∈ propertynames(obj)
                method = Symbol(getproperty(obj, symbol))
            end
        end
    end
    return propagate_objective(
        obj,
        tlist,
        Val(method);
        initial_state=initial_state,
        controls_map=controls_map,
        kwargs...
    )
end


#! format: off
function propagate_objective(obj, tlist, method::Symbol; initial_state,
                             controls_map=IdDict(), kwargs...)
    return propagate_objective(obj, tlist, Val(method);
                               initial_state=initial_state,
                               controls_map=controls_map, kwargs...)
end


function propagate_objective(obj, tlist, method::Val; initial_state,
                             controls_map=IdDict(), kwargs...)
    wrk = initobjpropwrk(obj, tlist, method; initial_state=initial_state,
                         kwargs...)
    return _propagate_objective(
        obj, tlist, wrk; initial_state=initial_state,
        controls_map=controls_map, kwargs...
    )

end
#! format: on


# `propagate_objective` backend (note `wrk` argument instead of `method`)
function _propagate_objective(
    obj,
    tlist,
    wrk;
    initial_state=obj.initial_state,
    controls_map=IdDict(),
    kwargs...
)
    genfunc = objective_genfunc(obj, tlist; controls_map=controls_map)
    return QuantumPropagators._propagate(initial_state, genfunc, tlist, wrk; kwargs...)
end


"""Propagate multiple objectives in parallel.

```julia
result = propagate_objectives(objectives, tlist; use_threads=true, kwargs...)
```

runs [`propagate_objective`](@ref) for every objective in `objectives`,
collects and returns a vector of results. The propagation happens in parallel
if `use_threads=true` (default). All keyword parameters are passed to
[`propagate_objective`](@ref), except that if `initial_state` is given, it must
be a vector of initial states, one for each objective. Likewise, to pass
pre-allocated storage arrays to `storage`, a vector of storage arrays must be
passed. A simple `storage=true` will still work to return a vector of storage
results.
"""
function propagate_objectives(
    objectives,
    tlist;
    use_threads=true,
    storage=nothing,
    initial_state=[obj.initial_state for obj in objectives],
    kwargs...
)
    result = Vector{Any}(undef, length(objectives))
    @threadsif use_threads for (k, obj) in collect(enumerate(objectives))
        if isnothing(storage) || (storage isa Bool)
            result[k] = propagate_objective(
                obj,
                tlist;
                storage=storage,
                initial_state=initial_state[k],
                kwargs...
            )
        else
            result[k] = propagate_objective(
                obj,
                tlist;
                storage=storage[k],
                initial_state=initial_state[k],
                kwargs...
            )
        end
    end
    return [result...]  # choose an atomatic eltype
end


"""
```julia
wrk = initobjpropwrk(obj, tlist, method; kwargs...)
```

initializes a workspace for the propagation of
an [`AbstractControlObjective`](@ref).
"""
function initobjpropwrk(
    obj::AbstractControlObjective,
    tlist,
    method::Val;
    initial_state=obj.initial_state,
    info_msg=nothing,
    verbose=false,
    kwargs...
)
    if verbose && !isnothing(info_msg)
        @info info_msg
    end
    controls = getcontrols(obj.generator)
    controlvals = [discretize(control, tlist) for control in controls]
    # We'll construct some examplary dynamical generators. This is mainly for
    # method=:cheby, but it's the "safe" thing to do for any method that uses
    # generators for initpropwrk, so we're using this as the default
    # implementation
    zero_vals =
        IdDict(control => zero(controlvals[i][1]) for (i, control) in enumerate(controls))
    G_zero = evalcontrols(obj.generator, zero_vals)
    max_vals =
        IdDict(control => maximum(controlvals[i]) for (i, control) in enumerate(controls))
    G_max = evalcontrols(obj.generator, max_vals)
    min_vals =
        IdDict(control => minimum(controlvals[i]) for (i, control) in enumerate(controls))
    G_min = evalcontrols(obj.generator, min_vals)
    return QuantumPropagators.initpropwrk(
        initial_state,
        tlist,
        method,
        G_zero,
        G_max,
        G_min;
        verbose=verbose,
        kwargs...
    )
end


function initobjpropwrk(
    obj::AbstractControlObjective,
    tlist,
    method::Symbol;
    initial_state=obj.initial_state,
    kwargs...
)
    return initobjpropwrk(obj, tlist, Val(method); initial_state=initial_state, kwargs...)
end


function initobjpropwrk(
    obj::AbstractControlObjective,
    tlist,
    method::Val{:newton};
    initial_state=obj.initial_state,
    kwargs...
)
    return QuantumPropagators.initpropwrk(initial_state, tlist, method; kwargs...)
end


function initobjpropwrk(
    obj::AbstractControlObjective,
    tlist,
    method::Val{:expprop};
    initial_state=obj.initial_state,
    kwargs...
)
    return QuantumPropagators.initpropwrk(initial_state, tlist, method; kwargs...)
end
