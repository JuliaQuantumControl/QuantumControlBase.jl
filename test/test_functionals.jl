using Test
using LinearAlgebra
using QuantumControlBase.Functionals
using QuantumControlBase.Functionals:
    chi_re!, chi_sm!, chi_ss!, make_zygote_gradient, make_zygote_chi, grad_J_T_sm!
using QuantumControlBase.TestUtils


N_HILBERT = 10
N = 4
L = 2
N_T = 50
PROBLEM = dummy_control_problem(; N=N_HILBERT, n_objectives=N, n_controls=L, n_steps=N_T)


@testset "zygote-gradients" begin

    # Test that Zygote gradients and analytical gradients give the same result

    objectives = PROBLEM.objectives
    G1 = zeros(L * N_T)
    G2 = zeros(L * N_T)
    τ = rand(N) .* exp.((1im * 2π) .* rand(N))
    ∇τ = [rand(L, N_T) .* exp.((1im * 2π) .* rand(L, N_T)) for k = 1:N]
    grad_J_T_sm_zyg! = make_gradient(J_T_sm, objectives; via=:tau, force_zygote=true)
    grad_J_T_sm!(G1, τ, ∇τ)
    grad_J_T_sm_zyg!(G2, τ, ∇τ)
    @test norm(G1 - G2) < 1e-15

end


@testset "chi-with-tau" begin

    # Test that the various chi routines give the same result whether they are
    # called with ϕ states or with τ values

    objectives = PROBLEM.objectives
    χ1 = [similar(obj.initial_state) for obj in objectives]
    χ2 = [similar(obj.initial_state) for obj in objectives]
    ϕ = [random_state_vector(N_HILBERT) for k = 1:N]
    τ = [obj.target_state ⋅ ϕ[k] for (k, obj) in enumerate(objectives)]

    chi_re!(χ1, ϕ, objectives)
    chi_re!(χ2, ϕ, objectives; τ=τ)
    @test maximum(norm.(χ1 .- χ2)) < 1e-15

    chi_sm!(χ1, ϕ, objectives)
    chi_sm!(χ2, ϕ, objectives; τ=τ)
    @test maximum(norm.(χ1 .- χ2)) < 1e-15

    chi_ss!(χ1, ϕ, objectives)
    chi_ss!(χ2, ϕ, objectives; τ=τ)
    @test maximum(norm.(χ1 .- χ2)) < 1e-15

end


@testset "zygote-chi" begin

    # Test that the zygote chi routines give the same results as the analytical
    # chi routines

    objectives = PROBLEM.objectives
    χ1 = [similar(obj.initial_state) for obj in objectives]
    χ2 = [similar(obj.initial_state) for obj in objectives]
    ϕ = [random_state_vector(N_HILBERT) for k = 1:N]
    τ = [obj.target_state ⋅ ϕ[k] for (k, obj) in enumerate(objectives)]

    chi_re!(χ1, ϕ, objectives)
    chi_re_zyg! = make_chi(J_T_re, objectives; force_zygote=true)
    chi_re_zyg!(χ2, ϕ, objectives; τ=τ)
    @test maximum(norm.(χ1 .- χ2)) < 1e-15

    chi_sm!(χ1, ϕ, objectives)
    chi_sm_zyg! = make_zygote_chi(J_T_sm, objectives)
    chi_sm_zyg!(χ2, ϕ, objectives; τ=τ)
    @test maximum(norm.(χ1 .- χ2)) < 1e-15

    chi_ss!(χ1, ϕ, objectives)
    chi_ss_zyg! = make_zygote_chi(J_T_ss, objectives)
    chi_ss_zyg!(χ2, ϕ, objectives; τ=τ)
    @test maximum(norm.(χ1 .- χ2)) < 1e-15

end


@testset "make-gradient" begin

    # Test that the routine returned by `make_gradient` gives the same result
    # as the Zygote gradient

    objectives = PROBLEM.objectives
    G1 = zeros(L * N_T)
    G2 = zeros(L * N_T)
    τ = rand(N) .* exp.((1im * 2π) .* rand(N))
    ∇τ = [rand(L, N_T) .* exp.((1im * 2π) .* rand(L, N_T)) for k = 1:N]

    for functional in (J_T_sm, J_T_re, J_T_ss)
        grad_auto! = make_gradient(functional, objectives; via=:tau)
        grad_zyg! = make_zygote_gradient(functional, objectives)
        grad_auto!(G1, τ, ∇τ)
        grad_zyg!(G2, τ, ∇τ)
        @test norm(G1 - G2) < 1e-15
    end

end


@testset "make-chi" begin

    # Test that the routine returned by `make_chi` gives the same result
    # as the Zygote chi

    objectives = PROBLEM.objectives
    χ1 = [similar(obj.initial_state) for obj in objectives]
    χ2 = [similar(obj.initial_state) for obj in objectives]
    ϕ = [random_state_vector(N_HILBERT) for k = 1:N]
    τ = [obj.target_state ⋅ ϕ[k] for (k, obj) in enumerate(objectives)]

    for functional in (J_T_sm, J_T_re, J_T_ss)
        chi_auto! = make_chi(functional, objectives)
        chi_zyg! = make_zygote_chi(functional, objectives)
        chi_auto!(χ1, ϕ, objectives; τ=τ)
        chi_zyg!(χ2, ϕ, objectives)
        @test maximum(norm.(χ1 .- χ2)) < 1e-15
    end

end


@testset "gradient-via-chi" begin

    # Test that gradients calculated via chi give the same results as gradients
    # calculated via tau

    problem = dummy_control_problem(; N=N_HILBERT, n_objectives=2, n_controls=1, n_steps=2)
    objectives = problem.objectives
    dt = problem.tlist[2] - problem.tlist[1]
    Ĥ₀ = Array(objectives[1].generator[1])
    Ĥ₁ = Array(objectives[1].generator[2][1])
    𝟘 = zeros(ComplexF64, N_HILBERT, N_HILBERT)
    ϵ = objectives[1].generator[2][2]
    Ψtgt = [obj.target_state for obj in objectives]
    χ = [similar(Ψ) for Ψ in Ψtgt]
    χ_from_τ = [similar(Ψ) for Ψ in Ψtgt]
    ∇τ_tgt = [zeros(ComplexF64, 1, 2) for k = 1:2]
    ∇τ_chi = [zeros(ComplexF64, 1, 2) for k = 1:2]
    G1 = zeros(2)
    G2 = zeros(2)

    # forward propagation of states
    Ψ0 = [obj.initial_state for obj in objectives]
    Ĥ1 = Ĥ₀ + ϵ[1] * Ĥ₁
    Ĥ2 = Ĥ₀ + ϵ[2] * Ĥ₁
    Ψ1 = [exp(-1im * Ĥ1 * dt) * Ψ for Ψ in Ψ0]
    Ψ2 = [exp(-1im * Ĥ2 * dt) * Ψ for Ψ in Ψ1]

    # overlap with target state
    τ_tgt = [Ψtgt[k] ⋅ Ψ2[k] for k = 1:2]

    # generators for gradient backward propagation
    G1⁺ = [Ĥ1'  Ĥ₁'; 𝟘  Ĥ1']
    G2⁺ = [Ĥ2'  Ĥ₁'; 𝟘  Ĥ2']

    # backward propagation of target states / gradients
    ξ̃2 = [[zeros(ComplexF64, N_HILBERT); Ψtgt[k]] for k = 1:2]
    ξ̃1 = [exp(1im * G2⁺ * dt) * ξ̃ for ξ̃ in ξ̃2]
    for k = 1:2
        ∇τ_tgt[k][1, 2] = ξ̃1[k][1:N_HILBERT] ⋅ Ψ1[k]
        ξ̃1[k][1:N_HILBERT] .= 0
    end
    ξ̃0 = [exp(1im * G1⁺ * dt) * ξ̃ for ξ̃ in ξ̃1]
    for k = 1:2
        ∇τ_tgt[k][1, 1] = ξ̃0[k][1:N_HILBERT] ⋅ Ψ0[k]
        ξ̃0[k][1:N_HILBERT] .= 0
    end

    for functional in (J_T_sm, J_T_re, J_T_ss)

        # boundary condition for χ-backward prop
        chi! = make_chi(functional, objectives)
        chi!(χ, Ψ2, objectives)
        chi!(χ_from_τ, Ψ2, objectives; τ=τ_tgt)
        @test maximum(norm.(χ .- χ_from_τ)) < 1e-15
        τ_chi = [χ[k] ⋅ Ψ2[k] for k = 1:2]

        # backward propagation of chi states / gradients
        χ̃2 = [[zeros(ComplexF64, N_HILBERT); χ[k]] for k = 1:2]
        χ̃1 = [exp(1im * G2⁺ * dt) * χ̃ for χ̃ in χ̃2]
        for k = 1:2
            ∇τ_chi[k][1, 2] = χ̃1[k][1:N_HILBERT] ⋅ Ψ1[k]
            χ̃1[k][1:N_HILBERT] .= 0
        end
        χ̃0 = [exp(1im * G1⁺ * dt) * χ̃ for χ̃ in χ̃1]
        for k = 1:2
            ∇τ_chi[k][1, 1] = χ̃0[k][1:N_HILBERT] ⋅ Ψ0[k]
            χ̃0[k][1:N_HILBERT] .= 0
        end

        grad_J_T_via_tau! = make_gradient(functional, objectives; via=:tau)
        grad_J_T_via_chi! = make_gradient(functional, objectives; via=:chi)
        @test grad_J_T_via_tau! ≢ grad_J_T_via_chi!
        grad_J_T_via_tau!(G1, τ_tgt, ∇τ_tgt)
        grad_J_T_via_chi!(G2, τ_chi, ∇τ_chi)
        @test norm(G1 - G2) < 1e-15

    end

end
