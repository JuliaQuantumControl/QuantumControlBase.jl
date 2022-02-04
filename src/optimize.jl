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

const _SAVENAME_AVAILABLE_KEYS = Set([
    :accesses,
    :allowedtypes,
    :connector,
    :digits,
    :equals,
    :expand,
    :ignores,
    :sigdigits,
    :sort,
    :val_to_string,
])
const DEFAULT_OPTIMIZATION_SAVENAME_KWARGS = Dict{Symbol,Any}()


# See @optimize_or_load for documentation –
# Only the macro version should be public!
function optimize_or_load(
    _filter,
    path,
    problem;
    method,
    filename::Union{Nothing,AbstractString}=nothing,
    suffix=((isnothing(filename) ? "jld2" : replace(splitext(filename)[2], "." => ""))),
    prefix="",
    tag::Bool=DrWatson.readenv("DRWATSON_TAG", DrWatson.istaggable(suffix)),
    gitpath=DrWatson.projectdir(),
    storepatch::Bool=DrWatson.readenv("DRWATSON_STOREPATCH", false),
    force=false,
    verbose=true,
    wsave_kwargs=Dict(),
    savename_kwargs=DEFAULT_OPTIMIZATION_SAVENAME_KWARGS,
    kwargs...
)

    # _filter is only for attaching metadata to the result

    if isnothing(filename)
        filename = optimization_savename(
            problem,
            method=method,
            suffix=suffix,
            prefix=prefix,
            savename_kwargs=savename_kwargs,
            kwargs...,
        )
    end

    data, file = DrWatson.produce_or_load(
        path,
        Dict();
        filename=filename,
        suffix=suffix,
        prefix=prefix,
        tag=tag,
        gitpath=gitpath,
        loadfile=true,
        storepatch=storepatch,
        force=force,
        verbose=verbose,
        wsave_kwargs=wsave_kwargs,
        savename_kwargs...
    ) do _
        result = optimize(problem; method=method, kwargs...)
        data = Dict("result" => result)
        if !isnothing(_filter)
            data = _filter(data)
        end
        return data
    end

    return data["result"], file

end

optimize_or_load(problem; kwargs...) = optimize_or_load(nothing, "", problem; kwargs...)
optimize_or_load(path, problem; kwargs...) =
    optimize_or_load(nothing, path, problem; kwargs...)


"""
Run [`optimize`](@ref) and store the result, or load the result if it exists.

```julia
result, file = @optimize_or_load(
    path="",
    problem;
    method=<method>,
    filename=nothing,
    suffix="jld2",
    prefix=DrWatson.default_prefix(config),
    tag=DrWatson.readenv("DRWATSON_TAG", true),
    gitpath=DrWatson.projectdir(),
    storepatch::Bool=DrWatson.readenv("DRWATSON_STOREPATCH", false),
    force=false,
    verbose=true,
    wsave_kwargs=Dict(),
    savename_kwargs=DEFAULT_OPTIMIZATION_SAVENAME_KWARGS,
    kwargs...
)
```

runs `result = optimize(problem; method=<method>, kwargs...)` and stores
`result` in an automatically named file inside `path`. The
automatic file name is determined by [`optimization_savename`](@ref) and can be
overriden by passing an explicit `filename`. The full path to the output file
(`joinpath(path, filename)`) is returned as `file`.

In addition to the `result`, the data in the output `file` may also contain
some metadata, e.g. "gitcommit" containing the git commit hash of the project
the produced the file, and "script" with the file name and line number
where `@optimize_or_load` was called, see [`load_optimization`](@ref).

If `file` already exists (and `force=false`), load the `result` from that file
instead of running the optimization.

The `@optimize_or_load` macro is intended to integrate well with the
[`DrWatson`](https://juliadynamics.github.io/DrWatson.jl/stable/) framework
for scientific projects and utilizes several configuration options and utility
functions from `DrWatson`, see below. Note that even though `DrWatson` is
recomended, you are not *required* to use if for your projects in order to use
`@optimize_or_load` or any other part of `QuantumControl`.

## I/O Keywords

The following keyword arguments determine where the result is stored and in
which format.

* `filename`: A file name to override the automatic file name. The `filename`
   should not contain slashes: use `path` for the folder where `filename`
   should be created.
* `suffix`, `prefix`, `savename_kwargs`: Parameters for
  [`optimization_savename`](@ref), which determines the automatic file name
* `tag`: Whether to record the current "gitcommit" as metadata alongside the
   optimization result, via
   [`DrWatson.tagsave`](https://juliadynamics.github.io/DrWatson.jl/stable/save/#DrWatson.tagsave).
   If not given explicitly, determine automatically from `suffix` or the
   extension of `filename`.
* `gitpath`, `storepatch`: Passed to `DrWatson.tagsave` if `tag` is `true`.
* `force`: If `true`, run and store the optimization regardless of whether
  `file` already exists.
* `verbose`: If `true`, print info about the process, if `file` does not exist.
* `wsave_kwargs`: Additional keyword arguments to pass to
  [`DrWatson.wsave`](https://juliadynamics.github.io/DrWatson.jl/stable/save/#Saving-Tools-1),
  e.g., to enable compression

All other keyword arguments are passed directly to [`optimize`](@ref).

## Related Functions

* [`optimization_savename`](@ref): Function that determines the automatic
  filename
* [`DrWatson.@produce_or_load`](https://juliadynamics.github.io/DrWatson.jl/stable/save/#DrWatson.@produce_or_load):
  The lower-level backend implementing the functionality of
  `@optimize_or_load`.
* [`load_optimization`](@ref): Function to load a file produced by
  `@optimize_or_load`
"""
macro optimize_or_load(path, problem, args...)
    args = Any[args...]
    # Keywords added after a `;` are moved to the front of the expression
    # that is passed to the macro. So instead of getting the path string
    # an Expr is passed.
    if path isa Expr && path.head == :parameters
        length(args) > 0 || return :(throw(
            MethodError(@optimize_or_load, $(esc(path)), $(esc(problem)), $(esc.(args)...)),
        ))
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
            $(esc(path)),
            $(esc(problem));
            $(esc.(DrWatson.convert_to_kw.(args))...)
        ) do data  # _filter
            # Extract the `gitpath` kw arg if it's there
            kws = ((; kwargs...) -> Dict(kwargs...))(
                $(esc.(DrWatson.convert_to_kw.(args))...),
            )
            gitpath = get(kws, :gitpath, DrWatson.projectdir())
            # Include the script tag with checking for the type of dict keys, etc.
            data = DrWatson.scripttag!(data, $s; gitpath=gitpath)
            return data
        end
    end
end


"""Determine an automatic filename for storing an optimization result.

```julia
file = optimization_savename(
    path="",
    problem;
    method=<method>,
    suffix="jld2",
    prefix="",
    savename_kwargs=DEFAULT_OPTIMIZATION_SAVENAME_KWARGS,
    kwargs...,
)
```

finds an appropriate automatic filename for the result of
`optimize(problem; method=<method>, kwargs...)`.

By default, the `file` has the structure
`<path>/<prefix>_<key1>=<value1>_..._<keyN>=<valueN>_method=<method>.jld2`
where the key-value pairs are a subset of the keyword arguments used to
instantiate `problem`, respectively the keyword arguments in `kwargs`. The
`prefix` is best used as a "name" for the optimization problem to ensure a
unique file name.

Which key-value pairs that are taken into account and the way they are
formatted can be customized via `savename_kwargs`.
See [`default_optimization_savename_kwargs`](@ref) for the supported options.
"""
function optimization_savename(
    path,
    problem;
    method,
    suffix="jld2",
    prefix="",
    savename_kwargs=DEFAULT_OPTIMIZATION_SAVENAME_KWARGS,
    kwargs...
)
    for k ∈ keys(savename_kwargs)
        if k ∉ _SAVENAME_AVAILABLE_KEYS
            throw(
                ArgumentError(
                    "'$k' is not a valid keyword argument for savename. Use one of $(join(_SAVENAME_AVAILABLE_KEYS, ", "))",
                ),
            )
        end
    end
    c = Saving.OptimizationConfig(problem, method, kwargs)
    filename = DrWatson.savename(prefix, c, suffix; savename_kwargs...)
    return joinpath(path, filename)
end

optimization_savename(problem; kwargs...) = optimization_savename("", problem; kwargs...)


"""Set the default `savename_kwargs` for [`optimization_savename`](@ref).

```julia
savename_kwargs = default_optimization_savename_kwargs(;kwargs...)
```

sets entries in the `DEFAULT_OPTIMIZATION_SAVENAME_KWARGS` used in
`optimization_savename` and thus determines the automatic name used to store
optimization results.

Use

```julia
default_optimization_savename_kwargs(reset=true)
```

to clear the settings from any previous call to
`default_optimization_savename_kwargs`.

The following keyword arguments are supported, cf. [`DrWatson.savename`]
(https://juliadynamics.github.io/DrWatson.jl/dev/name/#DrWatson.savename):

* `accesses` - List of strings indicating which fields (keys in `kwargs` of
  [`ControlProblem`](@ref) or `kwargs` of
  [`optimize`](@ref)/[`@optimize_or_load`](@ref)) can be included in the
  output filename. By default, all fields with values matching `allowedtypes`
  (excluding anonymous functions) are used.
* `allowedtypes` - List of types of values eligible to be included in the
  filename. Defaults to `[Real, String, Symbol, TimeType, Function]`
* `connector` - String used to separate key-value pairs in the output filename.
  Defaults to `"_"`.
* `digits` - Used in `round` when formatting numbers, if no custom
  `val_to_string`.
* `equals` - String used between keys and values. Defaults to `"="`.
* `ignores` - List of strings indicating which fields should be ignored.
* `sigdigits` - Used in `round` when formatting numbers, if no custom
  `val_to_string`.
* `sort` - Whether to sort the fields alphabetically (default). If `false`, the
  resulting file name my not be stable.
* `val_to_string` - Function to convert values to string.
"""
function default_optimization_savename_kwargs(; reset=false, kwargs...)
    # Note: default values are effectively set by the methods defined in
    # src/saving.jl
    if reset
        empty!(DEFAULT_OPTIMIZATION_SAVENAME_KWARGS)
    end
    for (k, v) ∈ kwargs
        if k ∉ _SAVENAME_AVAILABLE_KEYS
            throw(
                ArgumentError(
                    "'$k' is not a valid keyword argument for savename. Use one of $(join(_SAVENAME_AVAILABLE_KEYS, ", "))",
                ),
            )
        end
        DEFAULT_OPTIMIZATION_SAVENAME_KWARGS[k] = v
    end
    return copy(DEFAULT_OPTIMIZATION_SAVENAME_KWARGS)
end


"""Load a previously stored optimization.

```julia
result = load_optimization(filename; verbose=true, kwargs...)
```

recovers a `result` previously stored by [`@optimize_or_load`](@ref).

```julia
result, metadata = load_optimization(filename; return_metadata=true, kwargs...)
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
