using Test
using LinearAlgebra
using QuantumControl.Functionals:
    J_T_sm, J_T_re, J_T_ss, J_a_fluence, grad_J_a_fluence!, make_grad_J_a, make_chi
using QuantumControlTestUtils.DummyOptimization: dummy_control_problem
using QuantumControlTestUtils.RandomObjects: random_state_vector
using GRAPE: GrapeWrk
using StableRNGs: StableRNG
import Zygote
import FiniteDifferences
using QuantumControlBase: QuantumControlBase, _set_default_ad_framework
using IOCapture


N_HILBERT = 10
N = 4
L = 2
N_T = 50
RNG = StableRNG(4290326946)
PROBLEM =
    dummy_control_problem(; N=N_HILBERT, n_objectives=N, n_controls=L, n_steps=N_T, rng=RNG)


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
    χ7 = [similar(obj.initial_state) for obj in objectives]
    χ8 = [similar(obj.initial_state) for obj in objectives]
    ϕ = [random_state_vector(N_HILBERT; rng=RNG) for k = 1:N]
    τ = [obj.target_state ⋅ ϕ[k] for (k, obj) in enumerate(objectives)]

    for functional in (J_T_sm, J_T_re, J_T_ss)

        #!format: off
        chi_analytical! = make_chi(functional, objectives; mode=:analytic)
        chi_auto! = make_chi(functional, objectives)
        chi_zyg! = make_chi(functional, objectives; mode=:automatic, automatic=Zygote)
        chi_zyg_phi! = make_chi(functional, objectives; mode=:automatic, automatic=Zygote, via=:phi)
        chi_zyg_tau! = make_chi(functional, objectives; mode=:automatic, automatic=Zygote, via=:tau)
        chi_fdm! = make_chi(functional, objectives; mode=:automatic, automatic=FiniteDifferences)
        chi_fdm_phi! = make_chi(functional, objectives; mode=:automatic, automatic=FiniteDifferences, via=:phi)
        chi_fdm_tau! = make_chi(functional, objectives; mode=:automatic, automatic=FiniteDifferences, via=:tau)
        #!format: on

        chi_analytical!(χ1, ϕ, objectives; τ)
        chi_auto!(χ2, ϕ, objectives; τ)
        chi_zyg!(χ3, ϕ, objectives; τ)
        chi_zyg_phi!(χ4, ϕ, objectives; τ)
        chi_zyg_tau!(χ5, ϕ, objectives; τ)
        chi_fdm!(χ6, ϕ, objectives; τ)
        chi_fdm_phi!(χ7, ϕ, objectives; τ)
        chi_fdm_tau!(χ8, ϕ, objectives; τ)

        @test maximum(norm.(χ1 .- χ2)) < 1e-12
        @test maximum(norm.(χ1 .- χ3)) < 1e-12
        @test maximum(norm.(χ1 .- χ4)) < 1e-12
        @test maximum(norm.(χ1 .- χ5)) < 1e-12
        @test maximum(norm.(χ1 .- χ6)) < 1e-12
        @test maximum(norm.(χ1 .- χ7)) < 1e-12
        @test maximum(norm.(χ1 .- χ8)) < 1e-12

    end

end


@testset "make-grad-J_a" begin
    tlist = PROBLEM.tlist
    wrk = GrapeWrk(PROBLEM)
    pulsevals = wrk.pulsevals

    J_a_val = J_a_fluence(pulsevals, tlist)
    @test J_a_val > 0.0

    G1 = copy(wrk.grad_J_a)
    grad_J_a_fluence!(G1, pulsevals, tlist)

    grad_J_a_zygote! = make_grad_J_a(J_a_fluence, tlist; mode=:automatic, automatic=Zygote)
    @test grad_J_a_zygote! ≢ grad_J_a_fluence!
    G2 = copy(wrk.grad_J_a)
    grad_J_a_zygote!(G2, pulsevals, tlist)

    grad_J_a_fdm! =
        make_grad_J_a(J_a_fluence, tlist; mode=:automatic, automatic=FiniteDifferences)
    @test grad_J_a_fdm! ≢ grad_J_a_fluence!
    @test grad_J_a_fdm! ≢ grad_J_a_zygote!
    G3 = copy(wrk.grad_J_a)
    grad_J_a_fdm!(G3, pulsevals, tlist)

    @test 0.0 ≤ norm(G2 - G1) < 1e-12  # zygote can be exact
    @test 0.0 < norm(G3 - G1) < 1e-12  # fdm should not be exact
    @test 0.0 < norm(G3 - G2) < 1e-10

end


@testset "J_T without analytic derivative" begin

    _set_default_ad_framework(nothing; quiet=true)
    J_T(ϕ, objectives; tau=nothing, τ=tau) = 1.0

    objectives = PROBLEM.objectives

    capture = IOCapture.capture(rethrow=Union{}) do
        make_chi(J_T, objectives)
    end
    @test contains(capture.output, "fallback to mode=:automatic")
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        @test contains(capture.value.msg, "no default `automatic`")
    end

    _set_default_ad_framework(Zygote; quiet=true)
    capture = IOCapture.capture() do
        make_chi(J_T, objectives)
    end
    @test capture.value isa Function
    @test contains(capture.output, "fallback to mode=:automatic")
    @test contains(capture.output, "automatic with Zygote")

    capture = IOCapture.capture(rethrow=Union{}) do
        make_chi(J_T, objectives; mode=:analytic)
    end
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        @test contains(capture.value.msg, "no analytic gradient")
    end

    _set_default_ad_framework(nothing; quiet=true)

end


@testset "J_a without analytic derivative" begin

    _set_default_ad_framework(nothing; quiet=true)

    J_a(pulsvals, tlist) = 0.0
    tlist = [0.0, 1.0]

    capture = IOCapture.capture(rethrow=Union{}) do
        make_grad_J_a(J_a, tlist)
    end
    @test contains(capture.output, "fallback to mode=:automatic")
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        @test contains(capture.value.msg, "no default `automatic`")
    end

    _set_default_ad_framework(Zygote; quiet=true)
    capture = IOCapture.capture() do
        make_grad_J_a(J_a, tlist)
    end
    @test capture.value isa Function
    @test contains(capture.output, "fallback to mode=:automatic")
    @test contains(capture.output, "automatic with Zygote")

    capture = IOCapture.capture(rethrow=Union{}) do
        make_grad_J_a(J_a, tlist; mode=:analytic)
    end
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        @test contains(capture.value.msg, "no analytic gradient")
    end

    _set_default_ad_framework(nothing; quiet=true)

end


module UnsupportedADFramework end


@testset "Unsupported AD Framework (J_T)" begin

    _set_default_ad_framework(UnsupportedADFramework; quiet=true)
    @test QuantumControlBase.DEFAULT_AD_FRAMEWORK == :UnsupportedADFramework

    J_T(ϕ, objectives; tau=nothing, τ=tau) = 1.0
    objectives = PROBLEM.objectives

    capture = IOCapture.capture(rethrow=Union{}, passthrough=false) do
        make_chi(J_T, objectives)
    end
    @test contains(capture.output, "fallback to mode=:automatic")
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        msg = "no analytic gradient, and no automatic gradient"
        @test contains(capture.value.msg, msg)
    end

    capture = IOCapture.capture(rethrow=Union{}, passthrough=false) do
        make_chi(J_T, objectives; automatic=UnsupportedADFramework)
    end
    @test contains(capture.output, "fallback to mode=:automatic")
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        msg = "no analytic gradient, and no automatic gradient"
        @test contains(capture.value.msg, msg)
    end

    capture = IOCapture.capture(rethrow=Union{}, passthrough=false) do
        make_chi(J_T, objectives; mode=:automatic, automatic=UnsupportedADFramework)
    end
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        msg = ": no automatic gradient"
        @test contains(capture.value.msg, msg)
    end

    _set_default_ad_framework(nothing; quiet=true)
    @test QuantumControlBase.DEFAULT_AD_FRAMEWORK == :nothing

end


@testset "Unsupported AD Framework (J_a)" begin

    _set_default_ad_framework(UnsupportedADFramework; quiet=true)
    @test QuantumControlBase.DEFAULT_AD_FRAMEWORK == :UnsupportedADFramework

    J_a(pulsvals, tlist) = 0.0
    tlist = [0.0, 1.0]

    capture = IOCapture.capture(rethrow=Union{}, passthrough=false) do
        make_grad_J_a(J_a, tlist)
    end
    @test contains(capture.output, "fallback to mode=:automatic")
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        msg = "no analytic gradient, and no automatic gradient"
        @test contains(capture.value.msg, msg)
    end

    capture = IOCapture.capture(rethrow=Union{}, passthrough=false) do
        make_grad_J_a(J_a, tlist; automatic=UnsupportedADFramework)
    end
    @test contains(capture.output, "fallback to mode=:automatic")
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        msg = "no analytic gradient, and no automatic gradient"
        @test contains(capture.value.msg, msg)
    end

    capture = IOCapture.capture(rethrow=Union{}, passthrough=false) do
        make_grad_J_a(J_a, tlist; mode=:automatic, automatic=UnsupportedADFramework)
    end
    @test capture.value isa ErrorException
    if capture.value isa ErrorException
        msg = ": no automatic gradient"
        @test contains(capture.value.msg, msg)
    end

    _set_default_ad_framework(nothing; quiet=true)
    @test QuantumControlBase.DEFAULT_AD_FRAMEWORK == :nothing

end


@testset "invalid functional" begin

    _set_default_ad_framework(Zygote; quiet=true)

    J_T(ϕ, objectives) = 1.0  # no τ keyword argument
    objectives = PROBLEM.objectives
    @test_throws ErrorException begin
        IOCapture.capture() do
            make_chi(J_T, objectives)
        end
    end

    function J_T_xxx(ϕ, objectives; tau=nothing, τ=tau)
        throw(DomainError("XXX"))
    end

    @test_throws DomainError begin
        IOCapture.capture() do
            make_chi(J_T_xxx, objectives)
        end
    end

    @test_throws DomainError begin
        IOCapture.capture() do
            make_chi(J_T_xxx, objectives; mode=:automatic)
        end
    end

    function J_a_xxx(pulsevals, tlist)
        throw(DomainError("XXX"))
    end

    tlist = [0.0, 1.0]
    capture = IOCapture.capture() do
        make_grad_J_a(J_a_xxx, tlist)
    end
    grad_J_a = capture.value
    @test_throws DomainError begin
        grad_J_a(1, 1, tlist)
    end

    _set_default_ad_framework(nothing; quiet=true)

end
