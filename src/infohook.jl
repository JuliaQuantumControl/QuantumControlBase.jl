
"""Combine multiple `info_hook` functions.

```julia
chain_infohooks(funcs...)
```

combines `funcs` into a single Function that can be passes as `info_hook` to
[`ControlProblem`](@ref) or any `optimize`-function.

Each function in `func` must be a suitable `info_hook` by itself. This means
that it should receive the optimization workspace object as its first
positional parameter, then positional parameters specific to the optimization
method, and then an arbitrary number of data parameters. It must return either
`nothing` or a tuple of "info" objects (which will end up in the `records` field of
the optimization result).

When chaining infohooks, the `funcs` will be called in series, and the "info"
objects will be accumulated into a single result tuple. The combined results
from previous `funcs` will be given to the subsequent `funcs` as data
parameters. This allows for the infohooks in the chain to communicate.

The chain will return the final combined result tuple, or `nothing` if all
`funcs` return `nothing`.

!!! note

    When instantiating a [`ControlProblem`](@ref), any `info_hook` that is a
    tuple will be automatically processed with `chain_infohooks`. Thus,
    `chain_infohooks` rarely has to be invoked manually.
"""
function chain_infohooks(funcs...)

    function _info_hook(args...)
        res = Tuple([])
        for func in funcs
            res_f = func(args..., res...)
            if !isnothing(res_f)
                res = (res..., res_f...)
            end
        end
        if length(res) == 0
            return nothing
        else
            return res
        end
    end

    return _info_hook

end
