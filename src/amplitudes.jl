module Amplitudes

export LockedAmplitude, ShapedAmplitude, ParametrizedAmplitude

using QuantumPropagators.Generators: discretize_on_midpoints
import QuantumPropagators.Generators:
    getcontrols, substitute_controls, evalcontrols, getcontrolderiv
using ..PulseParametrizations: PulseParametrization, SquareParametrization


#### LockedAmplitude ##########################################################


"""A time-dependent amplitude that is not a control.

```julia
ampl = LockedAmplitude(shape)
```

wraps around `shape`, which must be either a vector of values defined on the
midpoints of a time grid or a callable `shape(t)`.

```julia
ampl = LockedAmplitude(shape, tlist)
```

discretizes `shape` to the midpoints of `tlist`.
"""
abstract type LockedAmplitude end


function LockedAmplitude(shape)
    if shape isa Vector{Float64}
        return LockedPulseAmplitude(shape)
    else
        return LockedContinuousAmplitude(shape)
    end
end


function LockedAmplitude(shape, tlist)
    return LockedPulseAmplitude(discretize_on_midpoints(shape, tlist))
end


function Base.show(io::IO, ampl::LockedAmplitude)
    print(io, "LockedAmplitude(::$(typeof(ampl.shape)))")
end


struct LockedPulseAmplitude <: LockedAmplitude
    shape::Vector{Float64}
end


Base.Array(ampl::LockedPulseAmplitude) = ampl.shape


struct LockedContinuousAmplitude <: LockedAmplitude

    shape

    function LockedContinuousAmplitude(shape)
        try
            S_t = shape(0.0)
        catch
            error("A LockedAmplitude shape must either be a vector of values or a callable")
        end
        return new(shape)
    end

end

(ampl::LockedContinuousAmplitude)(t::Float64) = ampl.shape(t)

getcontrols(ampl::LockedAmplitude) = ()

substitute_controls(ampl::LockedAmplitude, controls_map) = ampl

function evalcontrols(ampl::LockedPulseAmplitude, vals_dict, tlist, n)
    return ampl.shape[n]
end

function evalcontrols(ampl::LockedContinuousAmplitude, vals_dict, tlist, n)
    # It's technically possible to determine t from (tlist, n), but maybe we
    # shouldn't
    error("LockedAmplitude must be initialized with tlist")
end

getcontrolderiv(ampl::LockedAmplitude, control) =
    (control ≡ ampl.control) ? LockedAmplitude(ampl.shape) : 0.0


#### ControlAmplitude #########################################################


# An amplitude that has `control` as the first field
abstract type ControlAmplitude end

getcontrols(ampl::ControlAmplitude) = (ampl.control,)

function substitute_controls(ampl::CT, controls_map) where {CT<:ControlAmplitude}
    control = get(controls_map, ampl.control, ampl.control)
    CT(control, (getfield(ampl, field) for field in fieldnames(CT)[2:end])...)
end


#### ShapedAmplitude ##########################################################


"""Product of a fixed shape and a control.

```julia
ampl = ShapedAmplitude(control; shape=shape)
```

produces an amplitude ``a(t) = S(t) ϵ(t)``, where ``S(t)`` corresponds to
`shape` and ``ϵ(t)`` corresponds to `control`. Both `control` and `shape`
should be either a vector of values defined on the midpoints of a time grid or
a callable `control(t)`, respectively `shape(t)`. In the latter case, `ampl`
will also be callable.

```julia
ampl = ShapedAmplitude(control, tlist; shape=shape)
```

discretizes `control` and `shape` to the midpoints of `tlist`.
"""
abstract type ShapedAmplitude <: ControlAmplitude end

function ShapedAmplitude(control; shape)
    if (control isa Vector{Float64}) && (shape isa Vector{Float64})
        return ShapedPulseAmplitude(control, shape)
    else
        try
            ϵ_t = control(0.0)
        catch
            error(
                "A ShapedAmplitude control must either be a vector of values or a callable"
            )
        end
        try
            S_t = shape(0.0)
        catch
            error("A ShapedAmplitude shape must either be a vector of values or a callable")
        end
        return ShapedContinuousAmplitude(control, shape)
    end
end

function Base.show(io::IO, ampl::ShapedAmplitude)
    print(io, "ShapedAmplitude(::$(typeof(ampl.control)); shape::$(typeof(ampl.shape)))")
end

function ShapedAmplitude(control, tlist; shape)
    control = discretize_on_midpoints(control, tlist)
    shape = discretize_on_midpoints(shape, tlist)
    return ShapedPulseAmplitude(control, shape)
end

struct ShapedPulseAmplitude <: ShapedAmplitude
    control::Vector{Float64}
    shape::Vector{Float64}
end

Base.Array(ampl::ShapedPulseAmplitude) = ampl.control .* ampl.shape


struct ShapedContinuousAmplitude <: ShapedAmplitude
    control
    shape
end

(ampl::ShapedContinuousAmplitude)(t::Float64) = ampl.shape(t) * ampl.control(t)


function evalcontrols(ampl::ShapedPulseAmplitude, vals_dict, tlist, n)
    return ampl.shape[n] * vals_dict[ampl.control]
end

function evalcontrols(ampl::ShapedContinuousAmplitude, vals_dict, tlist, n)
    # It's technically possible to determind t from (tlist, n), but maybe we
    # shouldn't
    error("ShapedAmplitude must be initialized with tlist")
end

getcontrolderiv(ampl::ShapedAmplitude, control) =
    (control ≡ ampl.control) ? LockedAmplitude(ampl.shape) : 0.0


#### ParametrizedAmplitude ####################################################


"""An amplitude determined by a pulse parametrization.

That is, ``a(t) = a(ϵ(t))`` with a bijective mapping between the value of
``a(t)`` and ``ϵ(t)``, e.g. ``a(t) = ϵ^2(t)`` (a [`SquareParametrization`](@ref
SquareParametrization)). Optionally, the amplitude may be multiplied with an
additional shape function, cf. [`ShapedAmplitude`](@ref).


```julia
ampl = ParametrizedAmplitude(control; parametrization)
```

initilizes ``a(t) = a(ϵ(t)`` where ``ϵ(t)`` is the `control`, and the mandatory
keyword argument `parametrization` is a [`PulseParametrization`](@ref
PulseParametrization). The `control` must either be a vector of values
discretized to the midpoints of a time grid, or a callable `control(t)`.

```julia
ampl = ParametrizedAmplitude(control; parametrization, shape=shape)
```

initializes ``a(t) = S(t) a(ϵ(t))`` where ``S(t)`` is the given `shape`. It
must be a vector if `control` is a vector, or a callable `shape(t)` if
`control` is a callable.


```julia
ampl = ParametrizedAmplitude(control, tlist; parametrization, shape=shape)
```

discretizes `control` and `shape` (if given) to the midpoints of `tlist` before
initialization.


```julia
ampl = ParametrizedAmplitude(
    amplitude, tlist; parametrization, shape=shape, parametrize=true
)
```

initializes ``ã(t) = S(t) a(t)`` where ``a(t)`` is the input `amplitude`.
First, if `amplitude` is a callable `amplitude(t)`, it is discretized to the
midpoints of `tlist`. Then, a `control` ``ϵ(t)`` is calculated so that ``a(t) ≈
a(ϵ(t))``. Clippling may occur if the values in `amplitude` cannot represented
with the given `parametrization`. Lastly, `ParametrizedAmplitude(control;
parametrization, shape)` is initialized with the calculated `control`.

Note that the `tlist` keyword argument is required when `parametrize=true` is
given, even if `amplitude` is already a vector.
"""
abstract type ParametrizedAmplitude <: ControlAmplitude end

abstract type ShapedParametrizedAmplitude <: ParametrizedAmplitude end

function ParametrizedAmplitude(
    control;
    parametrization::PulseParametrization,
    shape=nothing
)
    if isnothing(shape)
        if control isa Vector{Float64}
            return ParametrizedPulseAmplitude(control, parametrization)
        else
            try
                S_t = shape(0.0)
            catch
                error(
                    "A ParametrizedAmplitude control must either be a vector of values or a callable"
                )
            end
            return ParametrizedContinousAmplitude(control, parametrization)
        end
    else
        if (control isa Vector{Float64}) && (shape isa Vector{Float64})
            return ShapedParametrizedPulseAmplitude(control, shape)
        else
            try
                ϵ_t = control(0.0)
            catch
                error(
                    "A ParametrizedAmplitude control must either be a vector of values or a callable"
                )
            end
            try
                S_t = shape(0.0)
            catch
                error(
                    "A ParametrizedAmplitude shape must either be a vector of values or a callable"
                )
            end
            return ShapedParametrizedContinuousAmplitude(control, shape)
        end
    end
end


function ParametrizedAmplitude(
    control,
    tlist;
    parametrization::PulseParametrization,
    shape=nothing,
    parameterize=false
)
    control = discretize_on_midpoints(control, tlist)
    if parameterize
        control = parametrization.epsilon_of_a.(control)
    end
    if !isnothing(shape)
        shape = discretize_on_midpoints(shape, tlist)
    end
    return ParametrizedAmplitude(control; parametrization, shape)
end

function Base.show(io::IO, ampl::ParametrizedAmplitude)
    print(
        io,
        "ParametrizedAmplitude(::$(typeof(ampl.control)); parametrization=$(ampl.parametrization))"
    )
end

function Base.show(io::IO, ampl::ShapedParametrizedAmplitude)
    print(
        io,
        "ParametrizedAmplitude(::$(typeof(ampl.control)); parametrization=$(ampl.parametrization), shape::$(typeof(ampl.shape)))"
    )
end

struct ParametrizedPulseAmplitude <: ParametrizedAmplitude
    control::Vector{Float64}
    parametrization::PulseParametrization
end

function Base.Array(ampl::ParametrizedPulseAmplitude)
    return ampl.parametrization.a_of_epsilon.(ampl.control)
end

struct ParametrizedContinuousAmplitude <: ParametrizedAmplitude
    control
    parametrization::PulseParametrization
end

struct ShapedParametrizedPulseAmplitude <: ShapedParametrizedAmplitude
    control::Vector{Float64}
    shape::Vector{Float64}
    parametrization::PulseParametrization
end

struct ShapedParametrizedContinuousAmplitude <: ShapedParametrizedAmplitude
    control
    shape
    parametrization::PulseParametrization
end


function evalcontrols(ampl::ParametrizedAmplitude, vals_dict, tlist, n)
    return ampl.parametrization.a_of_epsilon(vals_dict[ampl.control])
end

function evalcontrols(ampl::ShapedParametrizedPulseAmplitude, vals_dict, tlist, n)
    return ampl.shape[n] * ampl.parametrization.a_of_epsilon(vals_dict[ampl.control])
end

function Base.Array(ampl::ShapedParametrizedPulseAmplitude)
    return ampl.shape .* ampl.parametrization.a_of_epsilon.(ampl.control)
end

function evalcontrols(ampl::ShapedParametrizedContinuousAmplitude, vals_dict, tlist, n)
    # It's technically possible to determine t from (tlist, n), but maybe we
    # shouldn't
    error("ParametrizedAmplitude must be initialized with tlist")
end


function getcontrolderiv(ampl::ParametrizedAmplitude, control)
    if control ≡ ampl.control
        return ParametrizationDerivative(control, ampl.parametrization.da_deps_derivative)
    else
        return 0.0
    end
end

function getcontrolderiv(ampl::ShapedParametrizedPulseAmplitude, control)
    if control ≡ ampl.control
        return ShapedParametrizationPulseDerivative(
            control,
            ampl.parametrization.da_deps_derivative,
            ampl.shape
        )
    else
        return 0.0
    end
end

function getcontrolderiv(ampl::ShapedParametrizedContinuousAmplitude, control)
    if control ≡ ampl.control
        return ShapedParametrizationContinuousDerivative(
            control,
            ampl.parametrization.da_deps_derivative,
            ampl.shape
        )
    else
        return 0.0
    end
end

struct ParametrizationDerivative <: ControlAmplitude
    control
    func
end

struct ShapedParametrizationPulseDerivative <: ControlAmplitude
    control::Vector{Float64}
    func
    shape::Vector{Float64}
end

struct ShapedParametrizationContinuousDerivative <: ControlAmplitude
    control
    func
    shape
end

function evalcontrols(deriv::ParametrizationDerivative, vals_dict, _...)
    return deriv.func(vals_dict[deriv.control])
end

function evalcontrols(deriv::ShapedParametrizationPulseDerivative, vals_dict, tlist, n)
    return deriv.shape[n] * deriv.func(vals_dict[deriv.control])
end

function evalcontrols(deriv::ShapedParametrizationContinuousDerivative, vals_dict, tlist, n)
    # It's technically possible to determine t from (tlist, n), but maybe we
    # shouldn't
    error("ParametrizedAmplitude must be initialized with tlist")
end


end
