import Base

"""A full control problem with multiple objectives.

```julia
ControlProblem(
   objectives=<list of objectives>,
   pulse_options=<dict of controls to pulse options>,
   tlist=<time grid>,
   kwargs...
)
```

Note that the control problem can only be instantiated via keyword arguments.

The `objectives` are a list of [`Objective`](@ref) instances, each defining an
initial state, a dynamical generator for the evolution of the state, and
(optionally) a target for the evolution.

The `pulse_options` are a dictionary (`IdDict`) mapping controls that occur in
the `objectives` to properties specific to the control method.

The `tlist` is the time grid on which the time evolution of the initial states
of each objective should be propagated.

The remaining `kwargs` are keyword arguments that are passed directl to the
optimal control method. These typically include e.g. the optimization
functional.

The control problem is solved by finding a set of controls that simultaneously
fulfill all objectives.
"""
struct ControlProblem
    # TODO: specify types
    # TODO: `pulse_options` is not a good name
    objectives
    pulse_options
    tlist
    kwargs
    function ControlProblem(;objectives, pulse_options, tlist, kwargs...)
        new(objectives, pulse_options, tlist, kwargs)
    end
end


"""A single optimization objective.

```julia
Objective(
    initial_state=<intial state>,
    generator=<dynamical generator>,
    [target=<optional target state or specification>]
)

Note that the objective can only be instantiated via keyword arguments.
```
"""
struct Objective
    initial_state
    generator
    target
    function Objective(;initial_state, generator, target=nothing)
        new(initial_state, generator, target)
    end
end


# adjoint for the nested-tuple dynamical generator (e.g. `(H0, (H1, ϵ))`)
function dynamical_generator_adjoint(G::Tuple)
    result = []
    for part in G
        if isa(part, Tuple)
            push!(result, (Base.adjoint(part[1]), part[2]))
        else
            push!(result, Base.adjoint(part))
        end
    end
    return Tuple(result)
end

# fallback adjoint
dynamical_generator_adjoint(G) = Base.adjoint(G)


"""Adjoint of an objective."""
function Base.adjoint(obj::Objective)
    initial_state_adj = obj.initial_state
    try initial_state_adj = Base.adjoint(obj.initial_state) catch end
    generator_adj = dynamical_generator_adjoint(obj.generator)
    target_adj = obj.target
    try target_adj = Base.adjoint(obj.target) catch end
    return Objective(
        initial_state=initial_state_adj,
        generator=generator_adj,
        target=target_adj
    )
end
