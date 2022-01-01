module QuantumControlBase

include("controlproblem.jl")
export ControlProblem, Objective, WeightedObjective

include("propagate.jl")
export objective_genfunc, propagate_objective, initobjpropwrk

include("controls.jl")
export discretize, discretize_on_midpoints, evalcontrols, evalcontrols!
export get_control_parameters, getcontrols, getcontrolderiv, getcontrolderivs

include("gradgen.jl")
export TimeDependentGradGenerator, GradVector, GradGenerator, resetgradvec!

include("liouvillian.jl")
export liouvillian

include("infohook.jl")
export chain_infohooks

include("optimize.jl")
export optimize

# Submodules:

include("shapes.jl")
include("functionals.jl")
include("conditionalthreads.jl")
include("testutils.jl")

end
