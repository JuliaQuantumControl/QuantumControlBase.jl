import Base

"""Base class for a single optimization objective.

All objectives must have a field `initial_state` and a field `generator`, at
minimum.
"""
abstract type AbstractControlObjective end


"""Standard optimization objective.

```julia
Objective(;
    initial_state=<initial_state>,
    generator=<genenerator>,
    target_state=<target_state>
)
```

describes an optimization objective where the time evoluation of the given
`initial_state` under the given `generator` aims towards `target_state`. The
`generator` here is e.g. a time-dependent Hamiltonian or Liouvillian.

The most common control problems in quantum control, e.g. state-to-state
transitions or quantum gate implementations can be expressed by simultaneously
fulfilling multiple objectives of this type.

Note that the objective can only be instantiated via keyword arguments.
"""
struct Objective{ST, GT} <: AbstractControlObjective
    initial_state :: ST
    generator :: GT
    target_state :: ST
    function Objective(;initial_state::ST, generator::GT, target_state::ST) where {ST, GT}
        new{ST, GT}(initial_state, generator, target_state)
    end
end


"""Standard optimization objective with a weight.

```julia
WeightedObjective(;
    initial_state=<initial_state>,
    generator=<genenerator>,
    target_state=<target_state>,
    weight=<weight>
)
```

initializes a control objective like [`Objective`](@ref), but with an
additional `weight` parameter (a float generally between 0 and 1) that weights
the objective relative to other objectives that are part of the same control
problem.
"""
struct WeightedObjective{ST, GT} <: AbstractControlObjective
    initial_state :: ST
    generator :: GT
    target_state :: ST
    weight :: Float64
    function WeightedObjective(;initial_state::ST, generator::GT, target_state::ST, weight::Float64) where {ST, GT}
        new{ST, GT}(initial_state, generator, target_state, weight)
    end
end


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

The `objectives` are a list of [`AbstractControlObjective`](@ref) instances,
each defining an initial state and a dynamical generator for the evolution of
that state. Usually, the objective will also include a target state (see
[`Objective`](@ref)) and possibly a weight (see [`WeightedObjective`](@ref).

The `pulse_options` are a dictionary (`IdDict`) mapping controls that occur in
the `objectives` to properties specific to the control method.

The `tlist` is the time grid on which the time evolution of the initial states
of each objective should be propagated.

The remaining `kwargs` are keyword arguments that are passed directly to the
optimal control method. These typically include e.g. the optimization
functional.

The control problem is solved by finding a set of controls that simultaneously
fulfill all objectives.
"""
struct ControlProblem{OT<:AbstractControlObjective, OPT<:AbstractDict, KWT}
    # TODO: `pulse_options` is not a good name
    objectives :: Vector{OT}
    pulse_options :: OPT
    tlist :: Vector{Float64}
    kwargs :: KWT
    function ControlProblem(;objectives, pulse_options, tlist, kwargs...)
        new{typeof(objectives[1]), typeof(pulse_options), typeof(kwargs)}(
            objectives, pulse_options, tlist, kwargs
        )
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


"""
```julia
adjoint(objective)
```

Adjoint of a control objective. The adjoint objective contains the adjoint of
the dynamical generator `obj.generator`, and adjoints of the
`obj.initial_state` / `obj.target_state` if these exist and have an adjoint.
"""
function Base.adjoint(obj::Objective)
    initial_state_adj = obj.initial_state
    try initial_state_adj = Base.adjoint(obj.initial_state) catch end
    generator_adj = dynamical_generator_adjoint(obj.generator)
    target_adj = obj.target_state
    try target_adj = Base.adjoint(obj.target_state) catch end
    return Objective(
        initial_state=initial_state_adj,
        generator=generator_adj,
        target_state=target_adj
    )
end

function Base.adjoint(obj::WeightedObjective)
    initial_state_adj = obj.initial_state
    try initial_state_adj = Base.adjoint(obj.initial_state) catch end
    generator_adj = dynamical_generator_adjoint(obj.generator)
    target_adj = obj.target_state
    try target_adj = Base.adjoint(obj.target_state) catch end
    return WeightedObjective(
        initial_state=initial_state_adj,
        generator=generator_adj,
        target_state=target_adj,
        weight=obj.weight
    )
end
