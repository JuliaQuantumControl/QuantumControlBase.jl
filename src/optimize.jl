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

# See @optimize_or_load for documentation –
# Only the macro version should be public!
function optimize_or_load(
    _filter,
    file,
    problem;
    method,
    suffix="jld2",
    tag::Bool=DrWatson.readenv("DRWATSON_TAG", DrWatson.istaggable(suffix)),
    gitpath=DrWatson.projectdir(),
    storepatch::Bool=DrWatson.readenv("DRWATSON_STOREPATCH", false),
    force=false,
    verbose=get(problem.kwargs, :verbose, false),
    wsave_kwargs=Dict(),
    metadata=nothing,
    kwargs...
)

    # _filter is only for attaching metadata to the result

    (suffix |> startswith(".")) && (suffix = suffix[2:end])
    if ".$suffix" ≠ splitext(file)[2]
        @warn "$file suffix is not $suffix. Appending file extension"
        file = "$file.$suffix"
    end
    if isfile(file) && verbose
        if force
            @info "Ignoring existing $file (force)"
        else
            @info "Loading result from $file"
        end
    end

    data, file = DrWatson.produce_or_load(
        "",
        Dict();
        filename=file,
        suffix="",
        tag=tag,
        gitpath=gitpath,
        loadfile=true,
        storepatch=storepatch,
        force=force,
        verbose=verbose,
        wsave_kwargs=wsave_kwargs
    ) do _
        result = optimize(problem; method=method, verbose=verbose, kwargs...)
        data = Dict("result" => result)
        if !isnothing(_filter)
            data = _filter(data)
        end
        !isnothing(metadata) && merge!(data, metadata)
        return data
    end

    return data["result"]

end

optimize_or_load(file, problem; kwargs...) =
    optimize_or_load(nothing, file, problem; kwargs...)


# Given a list of macro arguments, push all keyword parameters to the end.
#
# A macro will receive keyword arguments after ";" as either the first or
# second argument (depending on whether the macro is invoked together with
# `do`). The `reorder_macro_kw_params` function reorders the arguments to put
# the keyword arguments at the end or the argument list, as if they had been
# separated from the positional arguments by a comma instead of a semicolon.
#
# # Example
#
# With
#
# ```
# macro mymacro(exs...)
#     @show exs
#     exs = reorder_macro_kw_params(exs)
#     @show exs
# end
# ```
#
# the `exs` in e.g. `@mymacro(1, 2; a=3, b)` will end up as
#
# ```
# (1, 2, :($(Expr(:kw, :a, 3))), :($(Expr(:kw, :b, :b))))
# ```
#
# instead of the original
#
# ```
# (:($(Expr(:parameters, :($(Expr(:kw, :a, 3))), :b))), 1, 2)
# ```
function reorder_macro_kw_params(exs)
    exs = Any[exs...]
    i = findfirst([(ex isa Expr && ex.head == :parameters) for ex in exs])
    if !isnothing(i)
        extra_kw_def = exs[i].args
        for ex in extra_kw_def
            push!(exs, ex isa Symbol ? Expr(:kw, ex, ex) : ex)
        end
        deleteat!(exs, i)
    end
    return Tuple(exs)
end


"""
Run [`optimize`](@ref) and store the result, or load the result if it exists.

```julia
result = @optimize_or_load(
    file,
    problem;
    method,
    suffix="jld2",
    tag=DrWatson.readenv("DRWATSON_TAG", true),
    gitpath=DrWatson.projectdir(),
    storepatch::Bool=DrWatson.readenv("DRWATSON_STOREPATCH", false),
    force=false,
    verbose=true,
    wsave_kwargs=Dict(),
    metadata=nothing,
    kwargs...
)
```

runs `result = optimize(problem; method, kwargs...)` and stores
`result` in `file`. Note that the `method` keyword argument is mandatory. In
addition to the `result`, the data in the output `file` may also contain some
metadata, e.g. (automatically) "gitcommit" containing the git commit hash of
the project that produced the file, and "script" with the file name and line
number where `@optimize_or_load` was called, see [`load_optimization`](@ref).
If `metadata` is given as a dict on input, the data it contains will be
included in the output file.

If `file` already exists (and `force=false`), load the `result` from that file
instead of running the optimization.

The `@optimize_or_load` macro is intended to integrate well with the
[`DrWatson`](https://juliadynamics.github.io/DrWatson.jl/stable/) framework
for scientific projects and utilizes several configuration options and utility
functions from `DrWatson`, see below. Note that even though `DrWatson` is
recomended, you are not *required* to use if for your projects in order to use
`@optimize_or_load` or any other part of `QuantumControl`.

## I/O Keywords

The following keyword arguments determine how the `result` is stored:

* `suffix`. File extension of `file`, determining the output data format (see
  [DrWatson Saving Tools](https://juliadynamics.github.io/DrWatson.jl/stable/save/)).
  If `file` does not end with the given extension, it will be appended.
* `tag`: Whether to record the current "gitcommit" as metadata alongside the
  optimization result, via
  [`DrWatson.tagsave`](https://juliadynamics.github.io/DrWatson.jl/stable/save/#DrWatson.tagsave).
  If not given explicitly, determine automatically from `suffix`.
* `gitpath`, `storepatch`: Passed to `DrWatson.tagsave` if `tag` is `true`.
* `force`: If `true`, run and store the optimization regardless of whether
  `file` already exists.
* `verbose`: If `true`, print info about the process
* `wsave_kwargs`: Additional keyword arguments to pass to
  [`DrWatson.wsave`](https://juliadynamics.github.io/DrWatson.jl/stable/save/#Saving-Tools-1),
  e.g., to enable compression

All other keyword arguments are passed directly to [`optimize`](@ref).

## Related Functions

* [`DrWatson.@produce_or_load`](https://juliadynamics.github.io/DrWatson.jl/stable/save/#DrWatson.@produce_or_load):
  The lower-level backend implementing the functionality of
  `@optimize_or_load`.
* [`load_optimization`](@ref): Function to load a file produced by
  `@optimize_or_load`
"""
macro optimize_or_load(exs...)
    exs = reorder_macro_kw_params(exs)
    exs = Any[exs...]
    _isa_kw = arg -> (arg isa Expr && (arg.head == :kw || arg.head == :(=)))
    if (length(exs) < 2) || _isa_kw(exs[1]) || _isa_kw(exs[2])
        @show exs
        error(
            "@optimize_or_load macro must receive `file` and `problem` as positional arguments"
        )
    end
    if (length(exs) > 2) && !_isa_kw(exs[3])
        @show exs
        error(
            "@optimize_or_load macro only takes two positional arguments (`file` and `problem`)"
        )
    end
    file = popfirst!(exs)
    problem = popfirst!(exs)
    # Save the source file name and line number of the calling line.
    s = QuoteNode(__source__)
    # Wrap the function f, such that the source can be saved in the data Dict.
    return quote
        optimize_or_load($(esc(file)), $(esc(problem)); $(esc.(exs)...)) do data # _filter
            # Extract the `gitpath` kw arg if it's there
            kws = ((; kwargs...) -> Dict(kwargs))(; $(esc.(exs)...))
            gitpath = get(kws, :gitpath, DrWatson.projectdir())
            # Include the script tag with checking for the type of dict keys, etc.
            data = DrWatson.scripttag!(data, $s; gitpath=gitpath)
            return data
        end
    end
end


"""Load a previously stored optimization.

```julia
result = load_optimization(file; verbose=true, kwargs...)
```

recovers a `result` previously stored by [`@optimize_or_load`](@ref).

```julia
result, metadata = load_optimization(file; return_metadata=true, kwargs...)
```

also obtains a metadata dict containing e.g., "gitcommit" or "script" depending
on the options to [`@optimize_or_load`](@ref).

Calling `load_optimization` with `verbose=true` (default) will show the
metadata after loading the file.
"""
function load_optimization(file; return_metadata=false, verbose=true, kwargs...)
    data = DrWatson.wload(file)
    result = data["result"]
    metadata = filter(kv -> (kv[1] != "result"), data)
    if verbose
        metadata_str = join(["  $key: $val" for (key, val) ∈ metadata], "\n")
        @info ("Loaded optimization result from $file\n" * metadata_str)
    end
    if return_metadata
        return result, metadata
    else
        return result
    end
end
