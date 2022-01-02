"""Evaluate `control` at every point of `tlist`.

```julia
values = discretize(control, tlist; via_midpoints=true)
```

discretizes the given `control` to a Vector of values defined on the points of
`tlist`.

If `control` is a function, it will will first be evaluated at the midpoint of
`tlist`, see [`discretize_on_midpoints`](@ref), and then the values on the
midpoints are converted to values on `tlist`. This discretization is more
stable than directly evaluationg the control function at the values of `tlist`,
and ensures that repeated round-trips between [`discretize`](@ref) and
[`discretize_on_midpoints`](@ref) can be done safely, see the note in the
documentation of [`discretize_on_midpoints`](@ref).

The latter can still be achieved by passing `via_midpoints=false`. While such a
direct discretization is suitable e.g. for plotting, but it is unsuitable
for round-trips between [`discretize`](@ref) and
[`discretize_on_midpoints`](@ref)  (constant controls on `tlist` may result in
a zig-zag on the intervals of `tlist`).

If `control` is a vector, it will be returned un-modified if it is of the same
length as `tlist`. Otherwise, `control` must have one less value than `tlist`,
and is assumed to be defined on the midpoins of `tlist`. In that case,
[`discretize`](@ref) acts as the inverse of [`discretize_on_midpoints`](@ref).
See [`discretize_on_midpoints`](@ref) for how control values on `tlist` and
control values on the intervals of `tlist` are related.
"""
function discretize(control::Function, tlist; via_midpoints=true)
    if via_midpoints
        vals_on_midpoints = discretize_on_midpoints(control, tlist)
        return discretize(vals_on_midpoints, tlist)
    else
        return [control(t) for t in tlist]
    end
end

function discretize(control::Vector, tlist)
    if length(control) == length(tlist)
        return control
    elseif length(control) == length(tlist) - 1
        # convert `control` on intervals to values on `tlist`
        # cf. pulse_onto_tlist in Python krotov package
        vals = zeros(eltype(control), length(control) + 1)
        vals[1] = control[1]
        vals[end] = control[end]
        for i in 2:length(vals)-1
            vals[i] = 0.5 * (control[i-1] + control[i])
        end
        return vals
    else
        throw(ArgumentError(
            "control array must be defined on intervals of tlist"
        ))
    end
end


@doc raw"""
Evaluate `control` at the midpoints of `tlist`.

```
values = discretize_on_midpoints(control, tlist)
```

discretizes the given `control` to a Vector of values on the midpoints of
`tlist`. Hence, the resulting `values` will contain one less value than
`tlist`.

If `control` is a vector of values defined on `tlist` (i.e., of the same length
as `tlist`), it will be converted to a vector of values on the intervals of
`tlist`. The value for the first and last "midpoint" will remain the original
values at the beginning and end of `tlist`, in order to ensure exact bounary
conditions. For all other midpoints, the value for that midpoint will be
calculated by "un-averaging".

For example, for a `control` and `tlist` of length 5, consider the following
diagram:

~~~
tlist index:       1   2   3   4   5
tlist:             ⋅   ⋅   ⋅   ⋅   ⋅   input values cᵢ (i ∈ 1..5)
                   |̂/ ̄ ̄ ̂\ / ̂\ / ̂ ̄ ̄\|̂
midpoints:         x     x   x     x   output values pᵢ (i ∈ 1..4)
midpoints index:   1     2   3     4
~~~

We will have ``p₁=c₁`` for the first value, ``p₄=c₅`` for the last value. For
all other points, the control values ``cᵢ = \frac{p_{i-1} + p_{i}}{2}`` are the
average of the values on the midpoints. This implies the "un-averaging" for the
midpoint values ``pᵢ = 2 c_{i} - p_{i-1}``.

!!! note

    An arbitrary input `control` array may not be compatible with the above
    averaging formula. In this case, the conversion will be "lossy"
    ([`discretize`](@ref) will not recover the original `control` array; the
    difference should be considered a "discretization error"). However, any
    *further* round-trip conversions between points and intervals are bijective
    and preserve the boundary conditions. In this case, the
    [`discretize_on_midpoints`](@ref) and [`discretize`](@ref) methods are each
    other's inverse. This also implies that for an optimal control procedure,
    it is safe to modify *midpoint* values. Modifying the the values on the
    time grid directly on the other hand may accumulate discretization errors.

If `control` is a vector of one less length than `tlist`, it will be returned
unchanged, under the assumption that the input is already properly discretized.

If `control` is a function, the function will be directly evaluated at the
midpoints marked as `x` in the above diagram..
"""
function discretize_on_midpoints(control::T, tlist) where T<:Function
    tlist_midpoints = zeros(eltype(tlist), length(tlist) - 1)
    tlist_midpoints[1] = tlist[1]
    tlist_midpoints[end] = tlist[end]
    for i in 2:length(tlist_midpoints) - 1
        dt = tlist[i+1] - tlist[i]
        tlist_midpoints[i] = tlist[i] + 0.5 * dt
    end
    return discretize(control, tlist_midpoints; via_midpoints=false)
end

function discretize_on_midpoints(control::Vector, tlist)
    if length(control) == length(tlist) - 1
        return control
    elseif length(control) == length(tlist)
        vals = zeros(eltype(control), length(control) - 1)
        vals[1] = control[1]
        vals[end] = control[end]
        for i in 2:length(vals)-1
            vals[i] = 2 * control[i] - vals[i-1]
        end
        return vals
    else
        throw(ArgumentError(
            "control array must be defined on the points of tlist"
        ))
    end
end



"""Extract a Tuple of controls.

```julia
controls = getcontrols(generator)
```

extracts the controls from a single dynamical generator.

```julia
controls = getcontrols(objectives)
```

extracts the controls from a list of objectives (i.e., from each objective's
`generator`)

In either case, controls that occur multiple times, either in a single
generator, or throughout the different objectives, will occur only once in the
result.

By default, assumes that any `generator` is a nested Tuple, e.g.
`(H0, (H1, ϵ1), (H2, ϵ2), ...)` and extracts (ϵ1, ϵ2)

Each control must be a valid argument for `discretize`.
"""
function getcontrols(generator::Tuple)
    controls = []
    slots_dict = IdDict()  # utilized as Set of controls we've seen
    FuncOrVector = Union{Function, Vector}
    for (i, part) in enumerate(generator)
        if isa(part, Tuple)
            control = part[2]
            control :: FuncOrVector  # assert correct type
            if control in keys(slots_dict)
                # We've seen this control before, so we just record the slot
                # where it is referenced
                push!(slots_dict[control], i)
            else
                push!(controls, control)
                slots_dict[control] = [i]
            end
        end
    end
    return Tuple(controls)
end


function getcontrols(objectives::Vector{T}) where {T<:AbstractControlObjective}
    controls = []
    seen_control = IdDict{Any, Bool}()
    for obj in objectives
        obj_controls = getcontrols(obj.generator)
        for control in obj_controls
            if haskey(seen_control, control)
                # skip: already seen
            else
                push!(controls, control)
                seen_control[control] = true
            end
        end
    end
    return Tuple(controls)
end


"""Replace the controls in `generator` with static values.

```julia
G = evalcontrols(generator, vals_dict)
```

replaces the time-dependent controls in `generator` with the values in
`vals_dict` and returns the static operator `G`.

The `vals_dict` is a dictionary (`IdDict`) mapping controls as returned by
`getcontrols(generator)` to values.
"""
function evalcontrols(generator::Tuple, vals_dict::D) where D<:AbstractDict
    if isa(generator[1], Tuple)
        control = generator[1][2]
        G = vals_dict[control] * generator[1][1]
    else
        G = generator[1]
    end
    for part in generator[2:end]
        if isa(part, Tuple)
            control = part[2]
            G += vals_dict[control] * part[1]
        else
            G += part
        end
    end
    return G
end


"""In-place version of [`evalcontrols`](@ref).

```julia
evalcontrols!(G, generator, vals_dict)
```

acts as [`evalcontrols`](@ref), but modifies `G` in-place.
"""
function evalcontrols!(G, generator::Tuple, vals_dict::D) where D<:AbstractDict
    if isa(generator[1], Tuple)
        control = generator[1][2]
        G .= vals_dict[control] * generator[1][1]
    else
        G .= generator[1]
    end
    for part in generator[2:end]
        if isa(part, Tuple)
            control = part[2]
            G .+= vals_dict[control] * part[1]
        else
            G .+= part
        end
    end
    return G
end


"""Get a vector of the derivatives of `generator` w.r.t. each control.

```julia
getcontrolderivs(generator, controls)
```

return as vector containing the derivative of `generator` with respect to each
control in `controls`. The elements of the vector are either `nothing` if
`generator` does not depend on that particular control, or a function `μ(α)`
that evaluates the derivative for a particular value of the control, see
[`getcontrolderiv`](@ref).
"""
function getcontrolderivs(generator, controls)
    controlderivs = Vector{Union{Function, Nothing}}(nothing, length(controls))
    for (i, control) in enumerate(controls)
        controlderivs[i] = getcontrolderiv(generator, control)
    end
    return controlderivs
end


@doc raw"""
Get the derivative of the generator ``G`` w.r.t. the control ``ϵ(t)``.

```julia
μ  = getcontrolderiv(generator, control)
```

returns `nothing` if the `generator` (Hamiltonian or Liouvillian) does not
depend on `control`, or a function `μ(v)` that evaluates

```math
μ(v) = \left.\frac{∂G}{∂ϵ(t)}\right\vert_{ϵ(t)=v}
```

otherwise. That is, a call `μ(v)` will return the static operator resulting
from evaluating the derivative of the dynamical generator ``G`` with respect to
the control filed ``ϵ(t)`` at a particular point in time where the control
field takes the value ``v``.

Note that for the common case of linear control terms, e.g., ``Ĥ = Ĥ_0 + \sum_l
ϵ_l(t) Ĥ_l``, the derivative ``∂Ĥ/∂ϵ_l(t)`` is simply the control Hamiltonian
``Ĥ_l``. Thus, the resulting function `μ` will simply return ``Ĥ_l``, ignoring
the argument `v`.
"""
function getcontrolderiv(generator::Tuple, control)
    control_generator = nothing
    for part in generator
        if isa(part, Tuple)
            if part[2] === control
                if isnothing(control_generator)
                    control_generator = part[1]
                else
                    control_generator += part[1]
                end
            end
        end
    end
    if isnothing(control_generator)
        return nothing
    else
        return v -> control_generator
    end
end


"""Collect all control parameters from the given objectives.

```julia
get_control_parameters(objectives...; kwargs...)
```

returns a vector of control parameters extracted from the controls in
`objectives`. This first extracts the controls with [`getcontrols`] and then
the "control parameters" from each control. What are the "control parameters"
are depends on the type of the control. For controls that are function ϵ(t),
the control parameters are the values of the control function on intervals of
the time grid. The time grid in the case must be passed as a keyword argument
`tlist`.

The control parameters for multiple controls will be concatenated in the
returned vector.
"""
function get_control_parameters(objectives...; kwargs...)
    controls = getcontrols(objectives)
    parameters = [get_control_parameters(control; kwargs...)
                  for control in controls]
    return vcat(parameters...)
end


"""
```julia
get_control_parameters(func; tlist, on_midpoints=true)
```

returns the control parameters of a control function by discretizing to the
time grid in `tlist`. By default, the discretization is on intervals
(midpoints) of the time grid. With `on_midpoints=false`, the discretization
will be on the actual points of the time grid.
"""
function get_control_parameters(func::Function; tlist, on_midpoints=true,
                                kwargs...)
    if on_midpoints
        return discretize_on_midpoints(func, tlist)
    else
        return discretize(func, tlist)
    end
end
