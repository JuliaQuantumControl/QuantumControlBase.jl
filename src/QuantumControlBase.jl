module QuantumControlBase

include("controlproblem.jl")
include("controls.jl")

export ControlProblem, Objective
export discretize, setcontrolvals!, getcontrols

end
