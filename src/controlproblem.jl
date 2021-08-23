import Base

"""A full control problem with multiple objectives.

```julia
ControlProblem(
   objectives=<list of objectives>,
   pulse_options=<dict of controls to pulse options>,
   tlist=<time grid>
)
```

Note that the control problem can only be instantiated via keyword arguments.
"""
struct ControlProblem
    objectives
    pulse_options
    tlist
    function ControlProblem(;objectives, pulse_options, tlist)
        new(objectives, pulse_options, tlist)
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
