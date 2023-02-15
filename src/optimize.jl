using QuantumPropagators.Controls: substitute

"""Optimize a quantum control problem.

```julia
result = optimize(problem; method=<method>, kwargs...)
```

optimizes towards a solution of given [`problem`](@ref ControlProblem) with the
given optimization `method`. Any keyword argument temporarily overrides the
corresponding keyword argument in `problem`.
"""
function optimize(problem::ControlProblem; method, kwargs...)
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
        return optimize(temp_problem, method)
    else
        return optimize(problem, method)
    end
end

optimize(problem::ControlProblem, method::Symbol) = optimize(problem, Val(method))
#
# Note: Methods *must* be defined in the various optimization packages as e.g.
#
#   optimize(problem, method::Val{:krotov}) = optimize_krotov(problem)
#
