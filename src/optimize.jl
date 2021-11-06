"""Optimize a quantum control problem.

```julia
opt_result = optimize(problem; method=<method>, kwargs...)
```

optimizes towards a solution of given `problem` with the given optimization
`method`. All keyword arguments update (overwrite) parameters in `problem`
"""
optimize(problem::ControlProblem; method, kwargs...) = optimize(problem, method; kwargs...)
optimize(problem::ControlProblem, method::Symbol; kwargs...) = optimize(problem, Val(method); kwargs...)
#
# Note: Methods *must* be defined in the various optimization packages as e.g.
#
#   optimize(problem, method::Val{:krotov}; kwargs...)
#
