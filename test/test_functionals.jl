using Test
using LinearAlgebra
using QuantumControlBase.Functionals
using QuantumControlBase.Functionals: chi_re!, chi_sm!, chi_ss!, make_analytic_chi
using QuantumControlBase.TestUtils


N_HILBERT = 10
N = 4
L = 2
N_T = 50
PROBLEM = dummy_control_problem(; N=N_HILBERT, n_objectives=N, n_controls=L, n_steps=N_T)


@testset "functionals-tau-no-tau" begin

    # Test that the various chi routines give the same result whether they are
    # called with ϕ states or with τ values

    objectives = PROBLEM.objectives
    χ1 = [similar(obj.initial_state) for obj in objectives]
    χ2 = [similar(obj.initial_state) for obj in objectives]
    ϕ = [random_state_vector(N_HILBERT) for k = 1:N]
    τ = [obj.target_state ⋅ ϕ[k] for (k, obj) in enumerate(objectives)]

    @test J_T_re(ϕ, objectives) ≈ J_T_re(nothing, objectives; τ)
    chi_re!(χ1, ϕ, objectives)
    chi_re!(χ2, ϕ, objectives; τ=τ)
    @test maximum(norm.(χ1 .- χ2)) < 1e-12

    @test J_T_sm(ϕ, objectives) ≈ J_T_sm(nothing, objectives; τ)
    chi_sm!(χ1, ϕ, objectives)
    chi_sm!(χ2, ϕ, objectives; τ=τ)
    @test maximum(norm.(χ1 .- χ2)) < 1e-12

    @test J_T_ss(ϕ, objectives) ≈ J_T_ss(nothing, objectives; τ)
    chi_ss!(χ1, ϕ, objectives)
    chi_ss!(χ2, ϕ, objectives; τ=τ)
    @test maximum(norm.(χ1 .- χ2)) < 1e-12

end


@testset "make-chi" begin

    # Test that the routine returned by `make_chi` gives the same result
    # as the Zygote chi

    objectives = PROBLEM.objectives
    χ1 = [similar(obj.initial_state) for obj in objectives]
    χ2 = [similar(obj.initial_state) for obj in objectives]
    χ3 = [similar(obj.initial_state) for obj in objectives]
    χ4 = [similar(obj.initial_state) for obj in objectives]
    χ5 = [similar(obj.initial_state) for obj in objectives]
    χ6 = [similar(obj.initial_state) for obj in objectives]
    ϕ = [random_state_vector(N_HILBERT) for k = 1:N]
    τ = [obj.target_state ⋅ ϕ[k] for (k, obj) in enumerate(objectives)]

    for functional in (J_T_sm, J_T_re, J_T_ss)

        chi_analytical! = make_analytic_chi(functional, objectives)
        chi_auto! = make_chi(functional, objectives)
        chi_zyg! = make_chi(functional, objectives; force_zygote=true)
        chi_zyg_phi! = make_chi(functional, objectives; force_zygote=true, via=:phi)
        chi_zyg_tau! = make_chi(functional, objectives; force_zygote=true, via=:tau)
        chi_fdm! = make_chi(functional, objectives; use_finite_differences=true)

        chi_analytical!(χ1, ϕ, objectives; τ)
        chi_auto!(χ2, ϕ, objectives; τ)
        chi_zyg!(χ3, ϕ, objectives; τ)
        chi_zyg_phi!(χ4, ϕ, objectives; τ)
        chi_zyg_tau!(χ5, ϕ, objectives; τ)
        chi_fdm!(χ6, ϕ, objectives; τ)

        @test maximum(norm.(χ1 .- χ2)) < 1e-12
        @test maximum(norm.(χ1 .- χ3)) < 1e-12
        @test maximum(norm.(χ1 .- χ4)) < 1e-12
        @test maximum(norm.(χ1 .- χ5)) < 1e-12
        @test maximum(norm.(χ1 .- χ6)) < 1e-12

    end

end
