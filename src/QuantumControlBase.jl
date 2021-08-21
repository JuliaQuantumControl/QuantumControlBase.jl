module QuantumControlBase

include("controlproblem.jl")
export ControlProblem, Objective

include("controls.jl")
export discretize, setcontrolvals!, getcontrols

include("shapes.jl")
export flattop, box, blackman

end
