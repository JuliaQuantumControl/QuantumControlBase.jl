module QuantumControlBase

include("controlproblem.jl")
export ControlProblem, Objective

include("propagate.jl")

include("controls.jl")
export discretize, discretize_on_midpoints, setcontrolvals, setcontrolvals!
export getcontrols

include("shapes.jl")
export flattop, box, blackman

end
