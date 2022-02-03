module Saving

using Dates: TimeType

import DrWatson
import ..ControlProblem


struct OptimizationConfig
    problem::ControlProblem
    method::Symbol
    kwargs::Dict{Symbol,Any}
end


function allaccess(c)
    no_anon_funcs((k, v)) = !((v isa Function) && contains(String(nameof(v)), "#"))
    all_keys = String["method"]
    for dict in (c.kwargs, c.problem.kwargs)
        for (k, v) ∈ dict
            key = String(k)
            if !((v isa Function) && contains(String(nameof(v)), "#"))
                push!(all_keys, key)
            end
        end
    end
    return all_keys
end

DrWatson.allaccess(c::OptimizationConfig) = allaccess(c)


function access(c, key::AbstractString)
    (key == "method") && (return String(c.method))
    key_sym = Symbol(key)
    if key_sym ∈ keys(c.kwargs)
        val = c.kwargs[key_sym]
    else
        val = c.problem.kwargs[key_sym]
    end
    return val
end

DrWatson.access(c::OptimizationConfig, key::AbstractString) = access(c, key)


const ALLOWED_TYPES = [Real, String, Symbol, TimeType, Function]

function default_allowed(c)
    return ALLOWED_TYPES
end

DrWatson.default_allowed(c::OptimizationConfig) = default_allowed(c)


end
