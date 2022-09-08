import Base
import QuantumPropagators

"""Base class for a single optimization objective.

All objectives must have a field `initial_state` and a field `generator`, at
minimum. Also, objectives must be able to be instantiated via keyword
arguments.
"""
abstract type AbstractControlObjective end


"""Standard optimization objective.

```julia
Objective(;
    initial_state=<initial_state>,
    generator=<generator>,
    target_state=<target_state>
)
```

describes an optimization objective where the time evaluation of the given
`initial_state` under the given `generator` aims towards `target_state`. The
`generator` here is e.g. a time-dependent Hamiltonian or Liouvillian.

The most common control problems in quantum control, e.g. state-to-state
transitions or quantum gate implementations can be expressed by simultaneously
fulfilling multiple objectives of this type.

Note that the objective can only be instantiated via keyword arguments.
"""
struct Objective{ST,GT} <: AbstractControlObjective
    initial_state::ST
    generator::GT
    target_state::ST
    function Objective(; initial_state::ST, generator::GT, target_state::ST) where {ST,GT}
        new{ST,GT}(initial_state, generator, target_state)
    end
end


"""Minmal optimization objective (initial state and dynamical generator only).

```julia
Objective(;
    initial_state=<initial_state>,
    generator=<generator>,
)
```

describes and optimization objective like the standard [`Objective`](@ref),
except for functionals that are not expressed with respect to some
`target_state`. Having only an `initial_state` and a `generator`, this is the
minimal data structure that is a valid instance of
[`AbstractControlObjective`](@ref).
"""
struct MinimalObjective{ST,GT} <: AbstractControlObjective
    initial_state::ST
    generator::GT
    function MinimalObjective(; initial_state::ST, generator::GT) where {ST,GT}
        new{ST,GT}(initial_state, generator)
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
struct WeightedObjective{ST,GT} <: AbstractControlObjective
    initial_state::ST
    generator::GT
    target_state::ST
    weight::Float64
    function WeightedObjective(;
        initial_state::ST,
        generator::GT,
        target_state::ST,
        weight::Float64
    ) where {ST,GT}
        new{ST,GT}(initial_state, generator, target_state, weight)
    end
end


"""A full control problem with multiple objectives.

```julia
ControlProblem(
   objectives=<list of objectives>,
   tlist=<time grid>,
   kwargs...
)
```

Note that the control problem can only be instantiated via keyword arguments.

The `objectives` are a list of [`AbstractControlObjective`](@ref) instances,
each defining an initial state and a dynamical generator for the evolution of
that state. Usually, the objective will also include a target state (see
[`Objective`](@ref)) and possibly a weight (see [`WeightedObjective`](@ref)).

The `tlist` is the time grid on which the time evolution of the initial states
of each objective should be propagated.

The remaining `kwargs` are keyword arguments that are passed directly to the
optimal control method. These typically include e.g. the optimization
functional.

The control problem is solved by finding a set of controls that simultaneously
fulfill all objectives.
"""
struct ControlProblem{OT<:AbstractControlObjective}
    objectives::Vector{OT}
    tlist::Vector{Float64}
    kwargs::Dict{Symbol,Any}
    function ControlProblem(; objectives, tlist, kwargs...)
        kwargs_dict = Dict{Symbol,Any}(kwargs)  # make the kwargs mutable
        new{eltype(objectives)}(objectives, tlist, kwargs_dict)
    end
end


function Base.copy(problem::ControlProblem)
    return ControlProblem(
        objectives=problem.objectives,
        tlist=problem.tlist;
        problem.kwargs...
    )
end


# adjoint for the nested-tuple dynamical generator (e.g. `(H0, (H1, ϵ))`)
function dynamical_generator_adjoint(G::Tuple)
    result = []
    for part in G
        # `copy` materializes the `adjoint` view, so we don't end up with
        # unnecessary `Adjoint{Matrix}` instead of Matrix, for example
        if isa(part, Tuple)
            push!(result, (copy(Base.adjoint(part[1])), part[2]))
        else
            push!(result, copy(Base.adjoint(part)))
        end
    end
    return Tuple(result)
end

# fallback adjoint
dynamical_generator_adjoint(G) = Base.adjoint(G)


"""Construct the adjoint of an optimization objective.

```julia
adjoint(objective)
```

Adjoint of a control objective. The adjoint objective contains the adjoint of
the dynamical generator `obj.generator`. All other fields contain a copy of the
original field value.

The primary purpose of this adjoint is to facilitate the backward propagation
under the adjoint generator that is central to gradient-based optimization
methods such as GRAPE and Krotov's method.
"""
function Base.adjoint(obj::AbstractControlObjective)
    fields = propertynames(obj)
    adjoints = Dict{Symbol,Any}()  # field => adjoint value
    for field ∈ fields
        if field == :generator
            # For the generator, the adjoint *must* be defined. GRAPE and
            # Krotov critically depend on this for the backward prop
            adjoints[field] = dynamical_generator_adjoint(obj.generator)
        else
            # Any other field, it doesn't really matter too much whether
            # we take the adjoint or not (none of the normal optimization
            # methods depend on anything but the generator being the adjoint)
            adj_value = getproperty(obj, field)
            try
                adjoints[field] = copy(adj_value)
            catch
                # `copy` isn't available e.g. for Strings
                adjoints[field] = adj_value
            end
        end
    end
    return typeof(obj).name.wrapper(; adjoints...)
end


"""
```julia
controls = getcontrols(objectives)
```

extracts the controls from a list of objectives (i.e., from each objective's
`generator`). Controls that occur multiple times in the different objectives
will occur only once in the result.
"""
function QuantumPropagators.Controls.getcontrols(
    objectives::Vector{T}
) where {T<:AbstractControlObjective}
    controls = []
    seen_control = IdDict{Any,Bool}()
    for obj in objectives
        obj_controls = QuantumPropagators.Controls.getcontrols(obj.generator)
        for control in obj_controls
            if haskey(seen_control, control)
                # skip: already seen
            else
                push!(controls, control)
                seen_control[control] = true
            end
        end
    end
    return Tuple(controls)
end
