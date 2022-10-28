module PulseParametrizations

export SquareParametrization,
    TanhParametrization,
    TanhSqParametrization,
    LogisticParametrization,
    LogisticSqParametrization


#! format: off


"""Specification for a "time-local" pulse parametrization.

The parametrization is given as a collection of three functions:

* ``a(ϵ(t))``
* ``ϵ(a(t))``
* ``∂a/∂ϵ`` as a function of ``ϵ(t)``.
"""
struct PulseParametrization
    name::String
    a_of_epsilon::Function
    epsilon_of_a::Function
    da_deps_derivative::Function
end


function Base.show(io::IO, p::PulseParametrization)
    print(io, p.name)
end


"""Parametrization a(t) = ϵ²(t), enforcing pulse values ``a(t) ≥ 0``."""
SquareParametrization() = PulseParametrization(
    "SquareParametrization()",
    ϵ -> begin # a_of_epsilon
        a = ϵ^2
    end,
    a -> begin # epsilon_of_a
        a = max(a, 0.0)
        ϵ = √a
    end,
    ϵ -> begin # da_deps_derivative
        ∂a╱∂ϵ = 2ϵ
    end
)


"""Parametrization with a tanh function that enforces `a_min < a(t) < a_max`.
"""
function TanhParametrization(a_min, a_max)

    Δ = a_max - a_min
    Σ = a_max + a_min
    aₚ = eps(1.0)  # 2⋅10⁻¹⁶ (machine precision)
    @assert a_max > a_min

    PulseParametrization(
        "TanhParametrization($a_min, $a_max)",
        ϵ -> begin # a_of_epsilon
            a = tanh(ϵ) * Δ / 2 + Σ / 2
        end,
        a -> begin # epsilon_of_a
            a = clamp(2a / Δ - Σ / Δ, -1 + aₚ, 1 - aₚ)
            ϵ = atanh(a)  # -18.4 < ϵ < 18.4
        end,
        ϵ -> begin # da_deps_derivative
            ∂a╱∂ϵ = (Δ / 2) * sech(ϵ)^2
        end
    )

end


"""Parametrization with a tanh² function that enforces `0 ≤ a(t) < a_max`.
"""
function TanhSqParametrization(a_max)

    aₚ = eps(1.0)  # 2⋅10⁻¹⁶ (machine precision)
    @assert a_max > 0

    PulseParametrization(
        "TanhSqParametrization($a_max)",
        ϵ -> begin # a_of_epsilon
            a = a_max * tanh(ϵ)^2
        end,
        a -> begin # epsilon_of_a
            a = clamp(a / a_max, 0, 1 - aₚ)
            ϵ = atanh(√a)
        end,
        ϵ -> begin # da_deps_derivative
            ∂a╱∂ϵ = 2a_max * tanh(ϵ) * sech(ϵ)^2
        end
    )

end


"""
Parametrization with a Logistic function that enforces `a_min < a(t) < a_max`.
"""
function LogisticParametrization(a_min, a_max; k=1.0)

    Δ = a_max - a_min
    a₀ = eps(0.0)  # 5⋅10⁻³²⁴
    @assert a_max > a_min

    PulseParametrization(
        "LogisticParametrization($a_max, $a_max; k=$k)",
        ϵ -> begin # a_of_epsilon
            a = Δ / (1 + exp(-k * ϵ)) + a_min
        end,
        a -> begin # epsilon_of_a
            a′ = a - a_min
            a = max(a′ / (Δ - a′), a₀)
            ϵ = log(a) / k
        end,
        ϵ -> begin # da_deps_derivative
            e⁻ᵏᵘ = exp(-k * ϵ)
            ∂a╱∂ϵ = Δ * k * e⁻ᵏᵘ / (1 + e⁻ᵏᵘ)^2
        end
    )

end


"""
Parametrization with a Logistic-Square function that enforces `0 ≤ a(t) < a_max`.
"""
function LogisticSqParametrization(a_max; k=1.0)

    a₀ = eps(0.0)  # 5⋅10⁻³²⁴
    @assert a_max > 0

    PulseParametrization(
        "LogisticSqParametrization($a_max; k=$k)",
        ϵ -> begin # a_of_epsilon
            a = a_max * (2 / (1 + exp(-k * ϵ)) - 1)^2
        end,
        a -> begin # epsilon_of_a
            ρ = clamp(a / a_max, 0.0, 1.0)
            a = clamp((2 / (√ρ + 1)) - 1, a₀, 1.0)
            ϵ = -log(a) / k
        end,
        ϵ -> begin # da_deps_derivative
            eᵏᵘ = exp(k * ϵ)
            ∂a╱∂ϵ = 4k * a_max * eᵏᵘ * (eᵏᵘ - 1) / (eᵏᵘ + 1)^3
        end
    )

end


#! format: on
end
