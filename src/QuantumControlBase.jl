module QuantumControlBase

include("controlproblem.jl")
export ControlProblem, Objective, WeightedObjective

include("propagate.jl")
export obj_genfunc

include("controls.jl")
export discretize, discretize_on_midpoints, evalcontrols, evalcontrols!
export get_control_parameters, getcontrols, getcontrolderiv, getcontrolderivs

include("liouvillian.jl")
export liouvillian

include("shapes.jl")
export flattop, box, blackman

include("functionals.jl")
export F_ss, J_T_ss, chi_ss!, F_sm, J_T_sm, chi_sm!, F_re, J_T_re, chi_re!

# Submodules:

include("conditionalthreads.jl")
include("testutils.jl")

end
