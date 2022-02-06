using Test
using LinearAlgebra
using QuantumPropagators: initpropwrk, propstep!
using QuantumPropagators.Newton
using QuantumControlBase:
    TimeDependentGradGenerator, GradGenerator, GradVector, resetgradvec!, evalcontrols
using QuantumControlBase.TestUtils
using Zygote

@testset "GradGenerator" begin

    N = 10  # size of Hilbert space
    ρ = 1.0  # spectral radius
    Ĥ₀ = random_hermitian_matrix(N, ρ)
    Ĥ₁ = random_hermitian_matrix(N, ρ)
    Ĥ₂ = random_hermitian_matrix(N, ρ)
    Zero = zeros(ComplexF64, N, N)
    ϵ₁ = t -> 1.0
    ϵ₂ = t -> 1.0
    Ĥ_of_t = (Ĥ₀, (Ĥ₁, ϵ₁), (Ĥ₂, ϵ₂))
    Ψ = random_state_vector(N)
    ψ_max = maximum(abs.(Ψ))
    Ψtgt = random_state_vector(N)
    𝕚 = 1im
    dt = 1.25

    vals_dict = IdDict(ϵ₁ => 1.0, ϵ₂ => 1.0)

    G̃_of_t = TimeDependentGradGenerator(Ĥ_of_t)
    G̃ = evalcontrols(G̃_of_t, vals_dict)
    Ĥ = evalcontrols(Ĥ_of_t, vals_dict)

    Û_Ψ = exp(-𝕚 * Ĥ * dt) * Ψ

    Ψ̃ = GradVector(Ψ, 2)
    # did the initialization work?
    @test norm(Ψ̃.state - Ψ) < 1e-14
    @test norm(Ψ̃.grad_states[1]) == 0.0
    @test norm(Ψ̃.grad_states[2]) == 0.0
    @test length(Ψ̃.grad_states) == 2

    @test maximum(abs.(Ψ)) == ψ_max  # is Ψ still exactly the same state?
    wrk = NewtonWrk(Ψ̃)
    newton!(Ψ̃, G̃, dt, wrk)
    Ψ̃_out = copy(Ψ̃)
    @test maximum(abs.(Ψ)) == ψ_max  # is Ψ still exactly the same state?
    @test norm(Ψ̃_out.state - Û_Ψ) < 1e-12  # Ψ̃_out.state correct?
    @test norm(Ψ̃_out.grad_states[1]) > 0
    @test norm(Ψ̃_out.grad_states[2]) > 0
    # did the copy work?
    @test norm(Ψ̃_out.state - Ψ̃.state) == 0.0
    @test norm(Ψ̃_out.grad_states[1] - Ψ̃.grad_states[1]) == 0.0
    @test norm(Ψ̃_out.grad_states[2] - Ψ̃.grad_states[2]) == 0.0

    # reset Ψ̃ to the original input (also to test the resetgradvec! method)
    resetgradvec!(Ψ̃, Ψ)
    # state should be the same as after previous initialization
    @test norm(Ψ̃.state - Ψ) < 1e-14
    @test norm(Ψ̃.grad_states[1]) == 0.0
    @test norm(Ψ̃.grad_states[2]) == 0.0

    @test norm(Ψ̃_out.state - Û_Ψ) < 1e-12  # Ψ̃_out.state still correct?

    ###########################################################################
    # Compare against explicit Grad-Gen
    # This checks whether the application of a GradGenerator to a GradVector an
    # all the linear-algebra methods are implemented correctly

    #! format: off
    G̃_full = vcat(hcat(Ĥ,    Zero, Ĥ₁),
                  hcat(Zero, Ĥ,    Ĥ₂),
                  hcat(Zero, Zero, Ĥ))

    Ψ̃_full = vcat(Ψ̃.grad_states[1],
                  Ψ̃.grad_states[2],
                  Ψ̃.state)
    #! format: on
    # proper initialization? grad_states should be zero
    @test norm(Ψ̃_full) == norm(Ψ̃.state) == norm(Ψ)

    Ψ̃_out_full = exp(-𝕚 * G̃_full * dt) * Ψ̃_full
    # propagation correct?
    @test norm(Ψ̃_out_full[2N+1:3N] - Û_Ψ) < 1e-12

    # do we get the same results as from newton?
    @test norm(Ψ̃_out_full[2N+1:3N] - Ψ̃_out.state) < 1e-12
    @test norm(Ψ̃_out_full[1:N] - Ψ̃_out.grad_states[1]) < 1e-12
    @test norm(Ψ̃_out_full[N+1:2N] - Ψ̃_out.grad_states[2]) < 1e-12

    ###########################################################################
    # Test custom expprop

    wrk_exp = initpropwrk(Ψ̃, [0, dt], Val(:expprop), G̃)
    propstep!(Ψ̃, G̃, dt, wrk_exp)
    Ψ̃_out_exp = copy(Ψ̃)
    @test norm(wrk_exp.Ψ_full - Ψ̃_full) ≈ 0.0
    @test norm(wrk_exp.G_full - G̃_full) ≈ 0.0
    @test norm(Ψ̃_out_exp - Ψ̃_out) < 1e-12
    resetgradvec!(Ψ̃, Ψ)


    ###########################################################################
    # Compare against explicit split-up Grad-Gens
    # This checks whether the extension of the Gradient-Vector to multiple
    # controls is correct (the literature generally only gives the expression
    # for a single control)

    #! format: off
    G̃_full1 = vcat(hcat(Ĥ,    Ĥ₁),
                   hcat(Zero, Ĥ))
    G̃_full2 = vcat(hcat(Ĥ,    Ĥ₂),
                   hcat(Zero, Ĥ))
    Ψ̃_full1 = vcat(Ψ̃.grad_states[1],
                   Ψ̃.state)
    #! format: on
    @test maximum(abs.(Ψ)) == ψ_max  # is Ψ still exactly the same state?
    @test norm(Ψ̃_full1) == norm(Ψ)  # initialization correct?
    Ψ̃_full2 = vcat(Ψ̃.grad_states[2], Ψ̃.state)
    @test norm(Ψ̃_full2) == norm(Ψ)  # initialization correct?
    Ψ̃_out_full1 = exp(-𝕚 * G̃_full1 * dt) * Ψ̃_full1
    Ψ̃_out_full2 = exp(-𝕚 * G̃_full2 * dt) * Ψ̃_full2

    # propagation correct?
    @test norm(Ψ̃_out_full1[N+1:2N] - Û_Ψ) < 1e-12
    @test norm(Ψ̃_out_full2[N+1:2N] - Û_Ψ) < 1e-12

    # do we get the same results as with the combined grad-gen?
    @test norm(Ψ̃_out_full1[1:N] - Ψ̃_out_full[1:N]) < 1e-12
    @test norm(Ψ̃_out_full2[1:N] - Ψ̃_out_full[N+1:2N]) < 1e-12
    @test norm(Ψ̃_out_full1[N+1:2N] - Ψ̃_out_full[2N+1:3N]) < 1e-12
    @test norm(Ψ̃_out_full2[N+1:2N] - Ψ̃_out_full[2N+1:3N]) < 1e-12

    ###########################################################################
    # Compare against Zygote
    # This checks whether the gradients are correct, albeit a bit indirectly
    # (since Zygote can only calculate the gradient for a scalar function, in
    # this case the square-modulus of the overlap with a target)

    F_sm(ϵ₁, ϵ₂) = abs(dot(Ψtgt, exp(-1im * (Ĥ₀ + ϵ₁ * Ĥ₁ + ϵ₂ * Ĥ₂) * dt) * Ψ))^2
    grad_zygote = collect(gradient(F_sm, 1.0, 1.0))

    @test norm(Ψ̃_out.state - Û_Ψ) < 1e-12  # still correct?
    τ = dot(Ψtgt, Û_Ψ)
    # `grad` is gradient of F_sm based on Newton-prop of GradGenerator
    # For ∂F/∂τ see Eq. (3.47) of Phd Thesis of Michael Goerz
    grad = [
        2 * real(conj(τ) * dot(Ψtgt, Ψ̃_out.grad_states[1])),
        2 * real(conj(τ) * dot(Ψtgt, Ψ̃_out.grad_states[2]))
    ]
    @test abs(grad_zygote[1] - grad[1]) < 1e-10
    @test abs(grad_zygote[2] - grad[2]) < 1e-10

    ###########################################################################
    # Compare against Taylor series
    # This checks the gradients more directly against an alternative method

    commutator(A, B) = A * B - B * A

    """Evaluate ∂/∂ϵ exp(-𝕚 Ĥ dt) via a Taylor expansion."""
    function U_grad(Ĥ, μ̂, dt)
        # See Eq. (14) in de Fouquieres et. al, JMR 212, 412 (2011)
        Û = exp(-𝕚 * Ĥ * dt)
        converged = false
        Ĉ = μ̂
        terms = [(-𝕚 * dt) * Ĉ]
        n = 2
        while !converged
            Ĉ = commutator(Ĥ, Ĉ)
            term = -((𝕚 * dt)^n / factorial(big(n))) * Ĉ
            push!(terms, term)
            converged = (norm(term) < 1e-12)
            n += 1
        end
        return Û * sum(terms)
    end

    grad_taylor = [U_grad(Ĥ, Ĥ₁, dt) * Ψ, U_grad(Ĥ, Ĥ₂, dt) * Ψ]

    @test norm(Ψ̃_out.grad_states[1] - grad_taylor[1]) < 1e-10
    @test norm(Ψ̃_out.grad_states[2] - grad_taylor[2]) < 1e-10

end
