"""Evaluate control at every point of `tlist`."""
function discretize(control, tlist; via_midpoints=False)
    # By default, `control` is callable.
    # Arrays a handled by multiple dispatch
    # TODO
end



"""Extract controls from and Objective.H.

Each control must be a valid argument for `discretize`.
"""
function getcontrols(H)
    # TODO
end


"""Evaluate H_out = H(vals)

`vals` is a vector of numbers, each number corresponding to one entry of
`getcontrols(H)`.

The resulting H_out must be a valid argument for the `propagator`.
"""
function setcontrolvals!(H_out, H, vals)
end
