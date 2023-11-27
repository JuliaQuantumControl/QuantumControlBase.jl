# Extension of QuantumPropagators.propagate for control objectives.

import QuantumPropagators
using QuantumPropagators.Controls: substitute, get_controls
# from conditionalthreads.jl: @threadsif


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
    throw(
        ArgumentError(
            "No propagation method from keywords $(repr(symbols)) of Objective with properties $(propertynames(obj)) or in optimization keyword arguments $(keys(kwargs))"
        )
    )
end


"""Propagate with the dynamical generator of a control objective.

```julia
propagate_objective(
    obj,
    tlist;
    method,  # mandatory keyword argument
    initial_state=obj.initial_state,
    kwargs...
)
```

propagates `initial_state` under the dynamics described by `obj.generator`.

If `obj` has a property/field `prop_method` or `fw_prop_method`, its value will
be used as the default for `method` instead of :auto. An explicit keyword
argument for `method` always overrides the default.

All other `kwargs` are forwarded to the underlying
[`QuantumPropagators.propagate`](@ref) method for `obj.initial_state`.
"""
function propagate_objective(obj, tlist; method, initial_state=obj.initial_state, kwargs...)
    # TODO: get propagation method (and other options) from the properties of
    # `obj`.
    return QuantumPropagators.propagate(
        initial_state,
        obj.generator,
        tlist;
        method,
        kwargs...
    )
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
    return [result...]  # chooses an automatic eltype
end
