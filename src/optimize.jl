"""Optimize a quantum control problem.

```julia
opt_result = optimize(problem; method=<method>, kwargs...)
```

optimizes towards a solution of given [`problem`](@ref ControlProblem) with the
given optimization `method`. Any keyword argument temporarily overrides the
corresponding keyword argument in `problem`.
"""
function optimize(problem::ControlProblem; method, kwargs...)
    if length(kwargs) > 0
        tempproblem = copy(problem)
        merge!(tempproblem.kwargs, kwargs)
        return optimize(tempproblem, method)
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
