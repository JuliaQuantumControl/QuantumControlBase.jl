module QuantumControlBase

include("conditionalthreads.jl")   # submodule ConditionalThreads

include("objectives.jl")
export ControlProblem, Objective

include("propagate.jl")
export propagate_objective, propagate_objectives

include("gradgen.jl")

include("infohook.jl")
export chain_infohooks


include("pulse_parametrizations.jl")  # submodule PulseParametrizations
include("amplitudes.jl")              # submodule Amplitudes
include("shapes.jl")                  # submodule Shapes
include("functionals.jl")             # submodule Functionals
include("weyl_chamber.jl")            # submodule WeylChamber
include("testutils.jl")               # submodule TestUtils


include("optimize.jl")
export optimize, @optimize_or_load, load_optimization

using .TestUtils: optimize_with_dummy_method
optimize(problem, method::Val{:dummymethod}) = optimize_with_dummy_method(problem)

end
