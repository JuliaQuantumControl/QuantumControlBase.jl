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


"""Adjoint of an objective."""
function Base.adjoint(obj::Objective)
    initial_state_adj = Base.adjoint(obj.initial_state)
    generator_adj = Base.adjoint(obj.generator)
    target_adj = obj.target
    try target_adj = Base.adjoint(obj.target) catch end
    return Objective(
        initial_state=initial_state_adj,
        generator=generator_adj,
        target=target_adj
    )
end
