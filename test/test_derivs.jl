using Test
using LinearAlgebra
using QuantumControlBase: QuantumControlBase, getcontrolderiv, getcontrolderivs
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


function QuantumControlBase.getcontrolderiv(a::MySquareAmpl, control)
    if control ≡ a.control
        return MyScaledAmpl(2.0, control)
    else
        return 0.0
    end
end


function QuantumPropagators.Generators.evalcontrols(a::MyScaledAmpl, vals_dict, args...)
    return a.c * evalcontrols(a.control, vals_dict, args...)
end


@testset "Standard getcontrolderivs" begin
    H₀ = random_hermitian_matrix(5, 1.0)
    H₁ = random_hermitian_matrix(5, 1.0)
    H₂ = random_hermitian_matrix(5, 1.0)
    ϵ₁ = t -> 1.0
    ϵ₂ = t -> 1.0
    H = (H₀, (H₁, ϵ₁), (H₂, ϵ₂))

    @test getcontrolderiv(ϵ₁, ϵ₁) == 1.0
    @test getcontrolderiv(ϵ₁, ϵ₂) == 0.0

    derivs = getcontrolderivs(H₀, (ϵ₁, ϵ₂))
    @test all(isnothing.(derivs))

    derivs = getcontrolderivs(H, (ϵ₁, ϵ₂))
    @test derivs[1] isa Matrix{ComplexF64}
    @test derivs[2] isa Matrix{ComplexF64}
    @test norm(derivs[1] - H₁) < 1e-14
    @test norm(derivs[2] - H₂) < 1e-14

    for deriv in derivs
        O = evalcontrols(deriv, IdDict(ϵ₁ => 1.1, ϵ₂ => 2.0))
        @test O ≡ deriv
    end

    @test isnothing(getcontrolderiv(H, t -> 3.0))

end


@testset "Nonlinear getcontrolderivs" begin

    H₀ = random_hermitian_matrix(5, 1.0)
    H₁ = random_hermitian_matrix(5, 1.0)
    H₂ = random_hermitian_matrix(5, 1.0)
    ϵ₁ = t -> 1.0
    ϵ₂ = t -> 1.0
    H = (H₀, (H₁, MySquareAmpl(ϵ₁)), (H₂, MySquareAmpl(ϵ₂)))

    derivs = getcontrolderivs(H, (ϵ₁, ϵ₂))
    @test derivs[1] isa Generator
    @test derivs[2] isa Generator
    @test derivs[1].ops[1] ≡ H₁
    @test _AT(derivs[1]) ≡ MyScaledAmpl

    O₁ = evalcontrols(derivs[1], IdDict(ϵ₁ => 1.1, ϵ₂ => 2.0))
    @test O₁ isa Operator
    @test length(O₁.ops) == length(O₁.coeffs) == 1
    @test O₁.ops[1] ≡ H₁
    @test O₁.coeffs[1] ≈ (2 * 1.1)

    O₂ = evalcontrols(derivs[2], IdDict(ϵ₁ => 1.1, ϵ₂ => 2.0))
    @test O₂ isa Operator
    @test length(O₂.ops) == length(O₂.coeffs) == 1
    @test O₂.ops[1] ≡ H₂
    @test O₂.coeffs[1] ≈ (2 * 2.0)

    @test isnothing(getcontrolderiv(H, t -> 3.0))

end
