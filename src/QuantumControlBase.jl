module QuantumControlBase

#=
# The export here is simply to indicate which symbols should be re-exported in
# QuantumControl
export ControlProblem, Trajectory, optimize, propagate_trajectory
export propagate_trajectories

include("atexit.jl")
include("conditionalthreads.jl")
include("trajectories.jl")
include("control_problem.jl")
include("propagate.jl")
include("derivs.jl")
include("functionals.jl")
include("callbacks.jl")
include("check_amplitude.jl")
include("check_generator.jl")
include("optimize.jl")
=#

function __init__()
    msg = "The QuantumControlBase package is obsolete. Its functionality has been integrated into the main QuantumControl package"
    @warn msg
end


end
