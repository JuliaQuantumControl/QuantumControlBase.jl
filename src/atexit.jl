using JLD2: jldopen

"""
Register a callback to dump a running optimization to disk on unexpected exit.

A long-running optimization routine may use

```julia
set_atexit_save_optimization(
    filename, result; msg_property=:messsage, msg="Abort: ATEXIT"
)
```

to register a callback that writes the given `result` object to the given
`filename` in JLD2 format in the event that the program terminates
unexpectedly. The idea is to avoid data loss if the user presses `CTRL-C` in a
non-interactive program (`SIGINT`), or if the process receives a `SIGTERM` from
an HPC scheduler because the process has reached its allocated runtime limit.
Note that the callback cannot protect against data loss in all possible
scenarios, e.g., a `SIGKILL` will terminate the program without giving the
callback a chance to run (as will yanking the power cord).

The optimization routine must remove the callback with

```julia
popfirst!(Base.atexit_hooks)
```

when it exits normally.

If `msg_property` is not `nothing`, the given `msg` string will be stored in
the corresponding property of the (mutable) `result` object before it is
written out.

The resulting JLD2 file is compatible with
[`QuantumControl.load_optimization`](https://juliaquantumcontrol.github.io/QuantumControl.jl/stable/api/quantum_control_reference/#QuantumControl.Workflows.load_optimization).
"""
function set_atexit_save_optimization(
    filename,
    result;
    msg_property=:messsage,
    msg="Abort: ATEXIT"
)

    function dump_on_exit()
        if !isnothing(msg_property)
            setproperty!(result, msg_property, msg)
        end
        jldopen(filename, "w") do data
            data["result"] = result
        end
    end

    # the callback might not have very much time to run, so it's best to
    # precompile and save a few seconds later on when it matters.
    precompile(dump_on_exit, ())

    atexit(dump_on_exit)

end
