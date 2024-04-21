using QuantumPropagators.Controls: substitute
using QuantumPropagators.Interfaces: check_state
# from ./check_generator.jl: check_generator


"""Optimize a quantum control problem.

```julia
result = optimize(problem; method, check=true, kwargs...)
```

optimizes towards a solution of given [`problem`](@ref ControlProblem) with
the given `method`, which should be a `Module` implementing the method, e.g.,

```julia
using Krotov
result = optimize(problem; method=Krotov)
```

Note that `method` is a mandatory keyword argument.

If `check` is true (default), the `initial_state` and `generator` of each
trajectory is checked with [`check_state`](@ref) and [`check_generator`](@ref).
Any other keyword argument temporarily overrides the corresponding keyword
argument in [`problem`](@ref ControlProblem). These arguments are available to
the optimizer, see each optimization package's documentation for details.

To obtain the documentation for which options a particular method uses, run,
e.g.,

```julia
? optimize(problem, ::Val{:Krotov})
```

where `:Krotov` is the name of the module implementing the method. The above is
also the method signature that a `Module` wishing to implement a control method
must define.

The returned `result` object is specific to the optimization method.
"""
function optimize(
    problem::ControlProblem;
    method::Union{Module,Symbol},
    check=true,
    for_expval=true, # undocumented
    for_mutable_operator=true,  # undocumented
    for_immutable_operator=true, # undocumented
    for_immutable_state=true, # undocumented
    for_mutable_state=true, # undocumented
    for_pwc=true,  # undocumented
    for_time_continuous=false,  # undocumented
    for_parameterization=false, # undocumented
    kwargs...
)

    if length(kwargs) > 0
        temp_kwargs = copy(problem.kwargs)
        merge!(temp_kwargs, kwargs)
        # We need to instantiate a new ControlProblem explicitly, so we get the
        # benefit of the custom constructor, e.g. for handling a tuple of
        # info-hooks.
        temp_problem = ControlProblem(;
            trajectories=problem.trajectories,
            tlist=problem.tlist,
            temp_kwargs...
        )
        problem = temp_problem
    end

    if check
        for (i, traj) in enumerate(problem.trajectories)
            if !check_state(traj.initial_state; for_immutable_state, for_mutable_state)
                error("The `initial_state` of trajectory $i is not valid")
            end
            if !check_generator(
                traj.generator;
                state=traj.initial_state,
                tlist=problem.tlist,
                for_mutable_operator,
                for_immutable_operator,
                for_immutable_state,
                for_mutable_state,
                for_expval,
                for_pwc,
                for_time_continuous,
                for_parameterization,
            )
                error("The `generator` of trajectory $i is not valid")
            end
        end
    end

    return optimize(problem, method)

end

optimize(problem::ControlProblem, method::Symbol) = optimize(problem, Val(method))
optimize(problem::ControlProblem, method::Module) = optimize(problem, Val(nameof(method)))
#
# Note: Methods *must* be defined in the various optimization packages as e.g.
#
#   optimize(problem, method::Val{:krotov}) = optimize_krotov(problem)
#
