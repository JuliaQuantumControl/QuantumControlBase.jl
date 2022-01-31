module Saving

using Dates: TimeType

import DrWatson
import ..ControlProblem

struct OptimizationConfig
    problem ::  ControlProblem
    method :: Symbol
    kwargs :: Dict{Symbol, Any}
end


function default_prefix(c)
    name = get(c.problem.kwargs, :name, nothing)
    if !isnothing(name)
        return "$(c.method)_$(c.name)"
    else
        return String(c.method)
    end
end

DrWatson.default_prefix(c::OptimizationConfig) = default_prefix(c)


function allaccess(c)
    no_anon_funcs((k, v)) = !(
        (v isa Function) && contains(String(nameof(v)), "#")
    )
    return collect(
        union(keys(filter(no_anon_funcs, c.kwargs)),
              keys(filter(no_anon_funcs, c.problem.kwargs))
        )
    )
end

DrWatson.allaccess(c::OptimizationConfig) = allaccess(c)


function allignore(c)
    return ("chi", )
end

DrWatson.allignore(c::OptimizationConfig) = allignore(c)


function access(c, key)
    (key ∈ keys(c.kwargs)) ? c.kwargs[key] : c.problem.kwargs[key]
end

DrWatson.access(c::OptimizationConfig, key) = access(c, key)


function default_allowed(c)
    return (Real, String, Symbol, TimeType, Function)
end

DrWatson.default_allowed(c::OptimizationConfig) = default_allowed(c)

end
