module QuantumControlBase

include("controlproblem.jl")
export ControlProblem, Objective, WeightedObjective

include("propagate.jl")
export objective_genfunc, propagate_objective, initobjpropwrk

include("controls.jl")
export discretize, discretize_on_midpoints, evalcontrols, evalcontrols!
export get_control_parameters, getcontrols, getcontrolderiv, getcontrolderivs
export get_tlist_midpoints

include("gradgen.jl")
export TimeDependentGradGenerator, GradVector, GradGenerator, resetgradvec!

include("liouvillian.jl")
export liouvillian

include("infohook.jl")
export chain_infohooks


include("shapes.jl")               # submodule Shapes
include("functionals.jl")          # submodule Functionals
include("conditionalthreads.jl")   # submodule ConditionalThreads
include("testutils.jl")            # submodule TestUtils
include("saving.jl")               # submodule Saving


include("optimize.jl")
export optimize, @optimize_or_load

end
