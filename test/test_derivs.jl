using Test
using LinearAlgebra
using QuantumControlBase: QuantumControlBase, get_control_deriv, get_control_derivs
using QuantumPropagators
using QuantumPropagators.Generators
using QuantumPropagators.Controls
using QuantumPropagators: Generator, Operator
using QuantumControlBase.TestUtils: random_hermitian_matrix


_AT(::Generator{OT,AT}) where {OT,AT} = AT


struct MySquareAmpl
    control::Function
end


struct MyScaledAmpl
    c::Number
    control::Function
end


function QuantumControlBase.get_control_deriv(a::MySquareAmpl, control)
    if control ≡ a.control
        return MyScaledAmpl(2.0, control)
    else
        return 0.0
    end
end


function QuantumPropagators.Controls.evaluate(a::MyScaledAmpl, args...; vals_dict=IdDict())
    return a.c * evaluate(a.control, args...; vals_dict)
end


@testset "Standard get_control_derivs" begin
    H₀ = random_hermitian_matrix(5, 1.0)
    H₁ = random_hermitian_matrix(5, 1.0)
    H₂ = random_hermitian_matrix(5, 1.0)
    ϵ₁ = t -> 1.0
    ϵ₂ = t -> 1.0
    H = (H₀, (H₁, ϵ₁), (H₂, ϵ₂))

    @test get_control_deriv(ϵ₁, ϵ₁) == 1.0
    @test get_control_deriv(ϵ₁, ϵ₂) == 0.0

    derivs = get_control_derivs(H₀, (ϵ₁, ϵ₂))
    @test all(isnothing.(derivs))

    derivs = get_control_derivs(H, (ϵ₁, ϵ₂))
    @test derivs[1] isa Matrix{ComplexF64}
    @test derivs[2] isa Matrix{ComplexF64}
    @test norm(derivs[1] - H₁) < 1e-14
    @test norm(derivs[2] - H₂) < 1e-14

    for deriv in derivs
        O = evaluate(deriv; vals_dict=IdDict(ϵ₁ => 1.1, ϵ₂ => 2.0))
        @test O ≡ deriv
    end

    @test isnothing(get_control_deriv(H, t -> 3.0))

end


@testset "Nonlinear get_control_derivs" begin

    H₀ = random_hermitian_matrix(5, 1.0)
    H₁ = random_hermitian_matrix(5, 1.0)
    H₂ = random_hermitian_matrix(5, 1.0)
    ϵ₁ = t -> 1.0
    ϵ₂ = t -> 1.0
    H = (H₀, (H₁, MySquareAmpl(ϵ₁)), (H₂, MySquareAmpl(ϵ₂)))

    derivs = get_control_derivs(H, (ϵ₁, ϵ₂))
    @test derivs[1] isa Generator
    @test derivs[2] isa Generator
    @test derivs[1].ops[1] ≡ H₁
    @test _AT(derivs[1]) ≡ MyScaledAmpl

    O₁ = evaluate(derivs[1]; vals_dict=IdDict(ϵ₁ => 1.1, ϵ₂ => 2.0))
    @test O₁ isa Operator
    @test length(O₁.ops) == length(O₁.coeffs) == 1
    @test O₁.ops[1] ≡ H₁
    @test O₁.coeffs[1] ≈ (2 * 1.1)

    O₂ = evaluate(derivs[2]; vals_dict=IdDict(ϵ₁ => 1.1, ϵ₂ => 2.0))
    @test O₂ isa Operator
    @test length(O₂.ops) == length(O₂.coeffs) == 1
    @test O₂.ops[1] ≡ H₂
    @test O₂.coeffs[1] ≈ (2 * 2.0)

    @test isnothing(get_control_deriv(H, t -> 3.0))

end
