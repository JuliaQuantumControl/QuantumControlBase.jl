import Base
import QuantumPropagators
using Printf

using QuantumPropagators.Generators: Generator, Operator
import QuantumPropagators.Controls: substitute

# TODO: consider using kwargs for init_prop, and document that feature.
"""Optimization objective.

```julia
Objective(;
    initial_state,
    generator,
    target_state=nothing,
    weight=1.0,
    kwargs...
)
```

describes an optimization objective that is tracked by the time evolution of
the given `initial_state` under the given `generator`, e.g., a time-dependent
Hamiltonian or Liouvillian. Each objective represents a single propagated state
on which an optimization functional may depend.

The most common control problems in quantum control (state-to-state, gate
optimization) require that the `initial_state` evolves into a `target_state`,
which should be given as a keyword argument.

An optimization functional usually depends on *multiple* forward-propagated
states (i.e., multiple `objectives`). Sometimes, it is useful to weight the
contributions of different `objectives` relative to each other, see, e.g.,
Goerz *et al*., New J. Phys. 16, 055012 (2014). To this end, a `weight` can be
attached to each `Objective` as an optional keyword argument.

Any other keyword arguments are available to a custom functional as properties
of the `Objective` .

Note that the `Objective` can only be instantiated via keyword arguments, with
`initial_state` and `generator` being the only two mandatory keyword arguments.
"""
struct Objective{ST,GT}
    initial_state::ST
    generator::GT
    target_state::Union{Nothing,ST}
    weight::Float64
    kwargs::Dict{Symbol,Any}

    function Objective(;
        initial_state::ST,
        generator::GT,
        target_state::Union{Nothing,ST}=nothing,
        weight=1.0,
        kwargs...
    ) where {ST,GT}
        new{ST,GT}(initial_state, generator, target_state, weight, kwargs)
    end

end


function Base.show(io::IO, obj::Objective{ST,GT}) where {ST,GT}
    print(io, "Objective{$ST, $GT}(…)")
end



function Base.propertynames(obj::Objective, private::Bool=false)
    return (
        :initial_state,
        :generator,
        :target_state,
        :weight,
        keys(getfield(obj, :kwargs))...
    )
end


function Base.setproperty!(obj::Objective, name::Symbol, value)
    error("setproperty!: immutable struct of type Objective cannot be changed")
end


function Base.getproperty(obj::Objective, name::Symbol)
    if name ≡ :initial_state
        return getfield(obj, :initial_state)
    elseif name ≡ :generator
        return getfield(obj, :generator)
    elseif name ≡ :target_state
        return getfield(obj, :target_state)
    elseif name ≡ :weight
        return getfield(obj, :weight)
    else
        kwargs = getfield(obj, :kwargs)
        return get(kwargs, name) do
            error("type Objective has no property $name")
        end
    end
end


"""
```julia
objective = substitute(objective::Objective, replacements)
objectives = substitute(objectives::Vector{Objective}, replacements)
```

recursively substitutes the `initial_state`, `generator`, and `target_state`.
"""
function substitute(objective::Objective, replacements)
    initial_state = substitute(objective.initial_state, replacements)
    ST = typeof(initial_state)
    generator = substitute(objective.generator, replacements)
    target_state::Union{Nothing,ST} = nothing
    if !isnothing(objective.target_state)
        target_state = substitute(objective.target_state, replacements)
    end
    weight = objective.weight
    kwargs = getfield(objective, :kwargs)
    return Objective(; initial_state, generator, target_state, weight, kwargs...)
end

function substitute(objectives::Vector{OT}, replacements) where {OT<:Objective}
    return [substitute(obj, replacements) for obj ∈ objectives]
end


"""A full control problem with multiple objectives.

```julia
ControlProblem(;
   objectives,
   tlist,
   kwargs...
)
```

Note that the control problem can only be instantiated via keyword arguments.

The `objectives` are a list of [`Objective`](@ref) instances,
each defining an initial state and a dynamical generator for the evolution of
that state. Usually, the objective will also include a target state (see
[`Objective`](@ref)) and possibly a weight.

The `tlist` is the time grid on which the time evolution of the initial states
of each objective should be propagated.

The remaining `kwargs` are keyword arguments that are passed directly to the
optimal control method. These typically include e.g. the optimization
functional.

The control problem is solved by finding a set of controls that simultaneously
fulfill all objectives.
"""
struct ControlProblem
    objectives::Vector{Objective}
    tlist::Vector{Float64}
    kwargs::Dict{Symbol,Any}
    function ControlProblem(; objectives, tlist, kwargs...)
        kwargs_dict = Dict{Symbol,Any}(kwargs)  # make the kwargs mutable
        new(objectives, tlist, kwargs_dict)
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

function dynamical_generator_adjoint(G::Generator)
    ops = [dynamical_generator_adjoint(op) for op in G.ops]
    return Generator(ops, G.amplitudes)
end

function dynamical_generator_adjoint(G::Operator)
    ops = [dynamical_generator_adjoint(op) for op in G.ops]
    coeffs = [Base.adjoint(c) for c in G.coeffs]
    return Operator(ops, G.coeffs)
end

# fallback adjoint
dynamical_generator_adjoint(G) = copy(Base.adjoint(G))


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
function Base.adjoint(obj::Objective)
    initial_state = obj.initial_state
    generator = dynamical_generator_adjoint(obj.generator)
    target_state = obj.target_state
    weight = obj.weight
    kwargs = getfield(obj, :kwargs)
    Objective(; initial_state, generator, target_state, weight, kwargs...)
end


"""
```julia
controls = get_controls(objectives)
```

extracts the controls from a list of objectives (i.e., from each objective's
`generator`). Controls that occur multiple times in the different objectives
will occur only once in the result.
"""
function QuantumPropagators.Generators.get_controls(objectives::Vector{<:Objective})
    controls = []
    seen_control = IdDict{Any,Bool}()
    for obj in objectives
        obj_controls = QuantumPropagators.Generators.get_controls(obj.generator)
        for control in obj_controls
            if !haskey(seen_control, control)
                push!(controls, control)
                seen_control[control] = true
            end
        end
    end
    return Tuple(controls)
end
