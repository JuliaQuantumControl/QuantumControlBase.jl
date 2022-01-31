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


import DrWatson

function optimize_or_load(_filter, path, problem;
        method,
        suffix="jld2",
        prefix= DrWatson.default_prefix(problem),
        tag::Bool=DrWatson.readenv("DRWATSON_TAG", DrWatson.istaggable(suffix)),
        gitpath=DrWatson.projectdir(),
        loadfile=true,
        storepatch::Bool=DrWatson.readenv("DRWATSON_STOREPATCH", false),
        force=false,
        verbose=true,
        wsave_kwargs=Dict(),
        filename::Union{Nothing, AbstractString}=nothing,
        savename_kwargs=Dict(),
        kwargs...
    )

    # _filter is only for attaching metadata to the result

    c = Saving.OptimizationConfig(problem, method, kwargs)

    data, file = DrWatson.produce_or_load(
        path, c;
        suffix=suffix, prefix=prefix, tag=tag, gitpath=gitpath,
        loadfile=loadfile, storepatch=storepatch, force=force, verbose=verbose,
        wsave_kwargs=wsave_kwargs, savename_kwargs...
    ) do c
        result = optimize(c.problem; method=c.method, c.kwargs...)
        data = Dict("result" => result)
        if !isnothing(_filter)
            data = _filter(data)
        end
        return data
    end

    return data["result"], file # TODO: attach metadata

end

optimize_or_load(problem; kwargs...) = optimize_or_load(nothing, "", problem; kwargs...)
optimize_or_load(path, problem; kwargs...) = optimize_or_load(nothing, path, problem; kwargs...)


macro optimize_or_load(path, problem, args...)
    args = Any[args...]
    # Keywords added after a `;` are moved to the front of the expression
    # that is passed to the macro. So instead of getting the path string
    # an Expr is passed.
    if path isa Expr && path.head == :parameters
        length(args) > 0 || return :(throw(MethodError(@optimize_or_load,$(esc(path)),$(esc(problem)),$(esc.(args)...))))
        extra_kw_def = path.args
        path = problem
        problem = popfirst!(args)
        append!(args, extra_kw_def)
    end
    # Save the source file name and line number of the calling line.
    s = QuoteNode(__source__)
    # Wrap the function f, such that the source can be saved in the data Dict.
    return quote
        optimize_or_load(
            $(esc(path)), $(esc(problem)); $(esc.(DrWatson.convert_to_kw.(args))...)
        ) do data  # _filter
            # Extract the `gitpath` kw arg if it's there
            kws = ((;kwargs...)->Dict(kwargs...))($(esc.(DrWatson.convert_to_kw.(args))...))
            gitpath = get(kws, :gitpath, DrWatson.projectdir())
            # Include the script tag with checking for the type of dict keys, etc.
            data = DrWatson.scripttag!(data, $s; gitpath = gitpath)
            return data
        end
    end
end
