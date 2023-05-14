using QuantumPropagators.Controls: substitute
using QuantumPropagators.Interfaces: check_state
# from ./check_generator.jl: check_generator


"""Optimize a quantum control problem.

```julia
result = optimize(problem; method=<method>, check=true, kwargs...)
```

optimizes towards a solution of given [`problem`](@ref ControlProblem) with the
given optimization `method`. Any keyword argument temporarily overrides the
corresponding keyword argument in `problem`.

If `check` is true (default), the `initial_state` and `generator` of each
objective is checked with [`check_state`](@ref) and [`check_generator`](@ref).
"""
function optimize(
    problem::ControlProblem;
    method::Symbol,
    check=true,
    for_expval=true, # undocumented
    for_immutable_state=true, # undocumented
    for_mutable_state=true, # undocumented
    kwargs...
)

    if length(kwargs) > 0
        temp_kwargs = copy(problem.kwargs)
        merge!(temp_kwargs, kwargs)
        # We need to instantiate a new ControlProblem explicitly, so we get the
        # benefit of the custom constructor, e.g. for handling a tuple of
        # info-hooks.
        temp_problem = ControlProblem(;
            objectives=problem.objectives,
            tlist=problem.tlist,
            temp_kwargs...
        )
        problem = temp_problem
    end

    if check
        for (i, obj) in enumerate(problem.objectives)
            if !check_state(obj.initial_state; for_immutable_state, for_mutable_state)
                error("The `initial_state` of objective $i is not valid")
            end
            if !check_generator(
                obj.generator;
                state=obj.initial_state,
                tlist=problem.tlist,
                for_immutable_state,
                for_mutable_state,
                for_expval
            )
                error("The `generator` of objective $i is not valid")
            end
        end
    end

    return optimize(problem, method)

end

optimize(problem::ControlProblem, method::Symbol) = optimize(problem, Val(method))
#
# Note: Methods *must* be defined in the various optimization packages as e.g.
#
#   optimize(problem, method::Val{:krotov}) = optimize_krotov(problem)
#
