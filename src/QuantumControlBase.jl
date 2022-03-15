module QuantumControlBase

include("conditionalthreads.jl")   # submodule ConditionalThreads

include("controlproblem.jl")
export ControlProblem, Objective, MinimalObjective, WeightedObjective

include("propagate.jl")
export propagate_objective, propagate_objectives

include("controls.jl")
export discretize, discretize_on_midpoints
export get_control_parameters, getcontrols
export get_tlist_midpoints

include("gradgen.jl")

include("liouvillian.jl")
export liouvillian

include("infohook.jl")
export chain_infohooks


include("shapes.jl")               # submodule Shapes
include("functionals.jl")          # submodule Functionals
include("testutils.jl")            # submodule TestUtils
include("saving.jl")               # submodule Saving


include("optimize.jl")
export optimize, @optimize_or_load, optimization_savename, load_optimization
export default_optimization_savename_kwargs

using .TestUtils: optimize_with_dummy_method
optimize(problem, method::Val{:dummymethod}) = optimize_with_dummy_method(problem)

end
