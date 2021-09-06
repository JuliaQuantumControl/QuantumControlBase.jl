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
    for (i, part) in enumerate(generator)
        if isa(part, Tuple)
            control = part[2]
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


function getcontrols(objectives::Vector{Objective{I, G, T}}) where {I, G, T}
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


"""Construct `G` by plugging values into a general generator.

```julia
G = setcontrolvals(generator, vals_dict)
setcontrolvals!(G, generator, vals_dict)
```

evaluates the *specific* dynamical generator `G` by plugging in values into the
general `generator` according to `vals_dict`.

The `vals_dict` is a dictionary (`IdDict`) mapping controls as returned by
`getcontrols(generator)` to values.
"""
function setcontrolvals(generator::Tuple, vals_dict::D) where D<:AbstractDict
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


"""In-place version of [`setcontrolvals`](@ref)."""
function setcontrolvals!(G, generator::Tuple, vals_dict::D) where D<:AbstractDict
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
