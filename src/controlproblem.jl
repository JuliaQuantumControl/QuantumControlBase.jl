
"""A full control problem with multiple objectives."""
struct ControlProblem
    objectives
    pulse_options
    tlist
end


"""A single optimization objective."""
struct Objective
    intial_state
    H
    target
end


adjoint(obj::Objective) = Objective(adjoint(initial_state), adjoint(H), adjoint(target))
