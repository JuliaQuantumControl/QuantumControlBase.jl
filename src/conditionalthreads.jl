using Base.Threads
using Base.Threads: threadid, threading_run

"""Conditionally apply multi-threading to `for` loops.

This is a variation on `Base.Threads.@threads` that adds a run-time boolean
flag to enable or disable threading. It is intended for *internal use* in
packages building on `QuantumControlBase`.

Usage:

```julia
using QuantumControlBase: @threadsif

function optimize(objectives; use_threads=true)
    @threadsif use_threads for k = 1:length(objectives)
    # ...
    end
end
```
"""
macro threadsif(cond, loop)
    if !(isa(loop, Expr) && loop.head === :for)
        throw(ArgumentError("@threadsif requires a `for` loop expression"))
    end
    if !(loop.args[1] isa Expr && loop.args[1].head === :(=))
        throw(ArgumentError("nested outer loops are not currently supported by @threadsif"))
    end
    quote
        if $(esc(cond))
            $(Threads._threadsfor(loop.args[1], loop.args[2], :static))
        else
            $(esc(loop))
        end
    end
end
