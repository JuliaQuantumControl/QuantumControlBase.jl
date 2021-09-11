using Test
using QuantumPropagators
using QuantumControlBase
using LinearAlgebra
using Distributions


@testset "TLS dissipation" begin

    function ket(i)
        Ψ = zeros(ComplexF64, 2)
        Ψ[i+1] = 1.0
        return Ψ
    end

    bra(i) = adjoint(ket(i))

    ketbra(i, j) = ket(i) * bra(j)

    γ₁ = 0.5
    γ₂ = 0.2

    Âᵧ₁ = √γ₁ * ketbra(0, 1)
    Âᵧ₂ = √(2γ₂) * ketbra(1, 1)

    Ψ₀ = (ket(0) + ket(1)) / √2
    ρ⃗₀ = reshape(Ψ₀ * Ψ₀', :)
    ℒ = Array(liouvillian(nothing, [Âᵧ₁,Âᵧ₂]; convention=:TDSE)[1])

    T = 1.0
    tlist = [0.0, T]

    ρ⃗_out = propagate(ρ⃗₀, (tlist, i; kwargs...) -> ℒ, tlist; method=:expprop)
    ρ_out = reshape(ρ⃗_out, 2, 2)

    ρ_expected = 0.5 * ComplexF64[
            (2-exp(-γ₁*T))    (exp(-(γ₁/2 + γ₂)*T));
        (exp(-(γ₁/2 + γ₂)*T))       (exp(-γ₁*T))
    ]


    @test abs(1 - tr(ρ_out)) < 1e-15 # total population
    @test abs(tr(ρ_out^2)) < 1.0  # purity
    @test norm(ρ_out - ρ_expected) < 1e-15

end


@testset "LvN" begin

    N = 100

    function random_hermitian_matrix(N, ρ)
        σ = 1/√N
        d = Normal(0.0, σ)
        X = rand(d, (N, N))
        return ρ * (X + X') / (2*√2)
    end

    function random_state_vector(N)
        Ψ = rand(N) .* exp.((2π * im) .* rand(N))
        Ψ ./= norm(Ψ)
        return Ψ
    end

    function ket(i)
        Ψ = zeros(ComplexF64, N)
        Ψ[i] = 1.0
        return Ψ
    end

    bra(i) = adjoint(ket(i))

    ketbra(i, j) = ket(i) * bra(j)

    Ĥ₀ = random_hermitian_matrix(N, 1)
    Ĥ₁ = random_hermitian_matrix(N, 0.1)

    H = (Ĥ₀, (Ĥ₁, t->1.0))
    Ĥ = H[1] + H[2][1] * H[2][2](0)

    Ψ₀ = random_state_vector(N)
    ρ₀ = Ψ₀ * Ψ₀'
    ρ⃗₀ = reshape(ρ₀, :)
    𝕚 = 1im

    ℒ_nodiss = liouvillian(Ĥ; convention=:LvN)[1]
    @test norm(𝕚 * (Ĥ * ρ₀ - ρ₀ * Ĥ) - reshape(ℒ_nodiss * ρ⃗₀, N, N)) < 1e-15

    ℒ_nodiss = liouvillian(Ĥ; convention=:TDSE)[1]
    @test norm((Ĥ * ρ₀ - ρ₀ * Ĥ) - reshape(ℒ_nodiss * ρ⃗₀, N, N)) < 1e-15

    γ₁ = 0.2
    decay_to_ground = [√γ₁ * ketbra(1, i) for i ∈ 2:N]

    γ₂ = 0.1
    dephasing = [√γ₂ * ketbra(i, i) for i ∈ 1:N]

    c_ops = (decay_to_ground..., dephasing...)
    L = liouvillian(H, c_ops; convention=:LvN)
    ℒ = L[1] + L[2][1] * L[2][2](0)

    L = liouvillian(H, c_ops; convention=:LvN)
    ℒ = L[1] + L[2][1] * L[2][2](0)

    ρ̇_LvN = (
        𝕚 * (Ĥ * ρ₀ - ρ₀ * Ĥ)
        + sum([
            (A * ρ₀ * A' - (A' * A * ρ₀)/2 - (ρ₀ * A' * A)/2)
            for A ∈ c_ops
        ])
    )

    ρ̇ = reshape(ℒ * ρ⃗₀, N, N)

    @test norm(ρ̇ - ρ̇_LvN) < 1e-15

end
