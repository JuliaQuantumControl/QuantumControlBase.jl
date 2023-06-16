using Test
using Logging: with_logger
using LinearAlgebra: I
using QuantumPropagators
using QuantumControlTestUtils: QuantumTestLogger
using QuantumControlTestUtils.RandomObjects: random_matrix, random_state_vector
using QuantumControlBase: check_generator, check_amplitude

import QuantumControlBase: get_control_deriv
import QuantumPropagators.Controls: get_controls, evaluate, evaluate!, substitute

struct InvalidGenerator
    control
end

# Define the methods required for propagation, but not the methods required
# for gradients
get_controls(G::InvalidGenerator) = (G.control,)
evaluate(G::InvalidGenerator, args...) = I(4)
evaluate!(op, G, args...) = op
substitute(G::InvalidGenerator, args...) = G


struct InvalidAmplitude
    control
end

get_controls(a::InvalidAmplitude) = (a.control,)
substitute(a::InvalidAmplitude, args...) = a
evaluate(a::InvalidAmplitude, args...; kwargs...) = evaluate(a.control, args...; kwargs...)


struct InvalidAmplitudeWrongDeriv
    control
end

get_controls(a::InvalidAmplitudeWrongDeriv) = (a.control,)
substitute(a::InvalidAmplitudeWrongDeriv, args...) = a
evaluate(a::InvalidAmplitudeWrongDeriv, args...; kwargs...) =
    evaluate(a.control, args...; kwargs...)
get_control_deriv(::InvalidAmplitudeWrongDeriv, control) = nothing


@testset "Invalid generator" begin

    state = ComplexF64[1, 0, 0, 0]
    tlist = collect(range(0, 10, length=101))

    generator = InvalidGenerator(t -> 1.0)

    @test QuantumPropagators.Interfaces.check_generator(generator; state, tlist)

    test_logger = QuantumTestLogger()
    with_logger(test_logger) do
        @test check_generator(generator; state, tlist) ≡ false
    end

    @test "`get_control_derivs(generator, controls)` must be defined" ∈ test_logger
    @test "`get_control_deriv(generator, control)` must return `nothing` if `control` is not in `get_controls(generator)`" ∈
          test_logger

end


@testset "Invalid amplitudes" begin

    N = 5
    state = random_state_vector(N)
    tlist = collect(range(0, 10, length=101))

    H₀ = random_matrix(5; hermitian=true)
    H₁ = random_matrix(5; hermitian=true)
    ampl = InvalidAmplitude(t -> 1.0)

    @test QuantumPropagators.Interfaces.check_amplitude(ampl; tlist)

    H = hamiltonian(H₀, (H₁, ampl))

    @test QuantumPropagators.Interfaces.check_generator(H; state, tlist)

    test_logger = QuantumTestLogger()
    with_logger(test_logger) do
        @test check_generator(H; state, tlist) ≡ false
    end
    @test "get_control_deriv(ampl, control) must be defined" ∈ test_logger

    ampl = InvalidAmplitudeWrongDeriv(t -> 1.0)
    test_logger = QuantumTestLogger()
    with_logger(test_logger) do
        check_amplitude(ampl; tlist)
    end
    @test "get_control_deriv(ampl, control) for  control 1 must return an object that evaluates to a Number" ∈
          test_logger
    @test "get_control_deriv(ampl, control) must return 0.0 if it does not depend on `control`" ∈
          test_logger

end
