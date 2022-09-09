using Test
using LinearAlgebra
using QuantumControlBase: Objective
using QuantumControlBase.Functionals
using QuantumControlBase.WeylChamber
using QuantumControlBase.TestUtils

⊗ = kron
const 𝕚 = 1im

function ket(i::Int64; N)
    Ψ = zeros(ComplexF64, N)
    Ψ[i+1] = 1
    return Ψ
end

function ket(i::Int64; N)
    Ψ = zeros(ComplexF64, N)
    Ψ[i+1] = 1
    return Ψ
end

function ket(indices::Int64...; N)
    Ψ = ket(indices[1]; N=N)
    for i in indices[2:end]
        Ψ = Ψ ⊗ ket(i; N=N)
    end
    return Ψ
end

function ket(label::AbstractString; N)
    indices = [parse(Int64, digit) for digit in label]
    return ket(indices...; N=N)
end

const CNOT = [
    1  0  0  0
    0  1  0  0
    0  0  0  1
    0  0  1  0
]

const CPHASE8 = [
    1  0  0  0
    0  1  0  0
    0  0  1  0
    0  0  0  exp(𝕚 * π / 8)
]

const sqrt_SWAP = [
    1      0        0      0
    0   1/2-𝕚/2  1/2+𝕚/2   0
    0   1/2+𝕚/2  1/2-𝕚/2   0
    0      0        0      1
]

const SWAP = sqrt_SWAP^2

@testset "find-gate" begin

    # This was originally going to test a find_gate routine, but that routine
    # turned out to be a very intuitive one-liner, so it was better not to
    # define it. Still, it's a good idea to test out the pattern of obtaining
    # transformed basis states and reconstructing the gate from the transformed
    # basis states.
    #
    # In particular, getting the transformed basis states is actually a little
    # bit un-intuitive: it is a matrix-vector product with the transpose gate!

    N = 10
    basis = [ket("00"; N), ket("01"; N), ket("10"; N), ket("11"; N)]

    #! format: off
    U = [
        #  00 01 10 11 :j
        1  0  0  0  # i: 00
        0  0  𝕚  0  # i: 01   # =  𝕚 |01⟩⟨10|  # U_01_10 = ⟨01|U|10⟩
        0 -𝕚  0  0  # i: 10   # = -𝕚 |10⟩⟨01|
        0  0  0  1  # i: 11
        # columns are image vectors
    ]
    #! format: on
    @test norm(U * U' - I) < 1e-15
    U_basis = [ket("00"; N), -𝕚 * ket("10"; N), 𝕚 * ket("01"; N), ket("11"; N)]
    U_basis2 = transpose(U) * basis
    @test maximum(norm.(U_basis - U_basis2)) < 1e-15
    gate = [basis[i] ⋅ U_basis[j] for i = 1:4, j = 1:4]
    @test norm(gate - U) < 1e-14

    U = exp(1im * random_hermitian_matrix(4, 1.0))
    U_basis = transpose(U) * basis
    gate = [basis[i] ⋅ U_basis[j] for i = 1:4, j = 1:4]
    @test norm(gate - U) < 1e-14

end

@testset "Local Invariants" begin

    g1, g2, g3 = local_invariants(CNOT)
    @test g1 ≈ 0 atol = 1e-15
    @test g2 ≈ 0 atol = 1e-15
    @test g3 ≈ 1 atol = 1e-15

    g1, g2, g3 = local_invariants(CPHASE8)
    @test g1 ≈ cos(0.5 * π / 8)^2 atol = 1e-15
    @test g2 ≈ 0 atol = 1e-15
    @test g3 ≈ (1 + 2 * g1) atol = 1e-15

    g1, g2, g3 = local_invariants(sqrt_SWAP)
    @test g1 ≈ 0 atol = 1e-15
    @test g2 ≈ (1 / 4) atol = 1e-15
    @test g3 ≈ 0 atol = 1e-15

    g1, g2, g3 = local_invariants(sqrt_SWAP')
    @test g1 ≈ 0 atol = 1e-15
    @test g2 ≈ (-1 / 4) atol = 1e-15
    @test g3 ≈ 0 atol = 1e-15

    g1, g2, g3 = local_invariants(SWAP)
    @test g1 ≈ -1 atol = 1e-15
    @test g2 ≈ 0 atol = 1e-15
    @test g3 ≈ -3 atol = 1e-15

end


@testset "Weyl Chamber Coordinates" begin

    c1, c2, c3 = weyl_chamber_coordinates(CNOT)
    @test c1 ≈ 0.5 atol = 1e-15
    @test c2 ≈ 0 atol = 1e-15
    @test c3 ≈ 0 atol = 1e-15
    Uc = canonical_gate(c1, c2, c3)
    @test norm(collect(weyl_chamber_coordinates(Uc)) - [c1, c2, c3]) < 1e-14

    c1, c2, c3 = weyl_chamber_coordinates(CPHASE8)
    @test c1 ≈ (1 / 16) atol = 1e-15  #  = γ / 2   for   γ = π/8
    @test c2 ≈ 0 atol = 1e-15
    @test c3 ≈ 0 atol = 1e-15
    # the Weyl chambers coordinates of the canonical gate are unstable: The
    # bottom surfaces of the W0 and W0* regions are equivalent, and floating
    # point errors can push π/8 to the equivalent 1-π/8

    c1, c2, c3 = weyl_chamber_coordinates(sqrt_SWAP)
    @test c1 ≈ (1 / 4) atol = 1e-15
    @test c2 ≈ (1 / 4) atol = 1e-15
    @test c3 ≈ (1 / 4) atol = 1e-15
    Uc = canonical_gate(c1, c2, c3)
    @test norm(collect(weyl_chamber_coordinates(Uc)) - [c1, c2, c3]) < 1e-14

    c1, c2, c3 = weyl_chamber_coordinates(sqrt_SWAP')
    @test c1 ≈ (3 / 4) atol = 1e-15
    @test c2 ≈ (1 / 4) atol = 1e-15
    @test c3 ≈ (1 / 4) atol = 1e-15
    Uc = canonical_gate(c1, c2, c3)
    @test norm(collect(weyl_chamber_coordinates(Uc)) - [c1, c2, c3]) < 1e-14

    c1, c2, c3 = weyl_chamber_coordinates(SWAP)
    @test c1 ≈ (1 / 2) atol = 1e-15
    @test c2 ≈ (1 / 2) atol = 1e-15
    @test c3 ≈ (1 / 2) atol = 1e-15
    Uc = canonical_gate(c1, c2, c3)
    @test norm(collect(weyl_chamber_coordinates(Uc)) - [c1, c2, c3]) < 1e-14

end


@testset "Gate Concurrence" begin
    @test gate_concurrence(CNOT) ≈ 1.0
    @test gate_concurrence(sqrt_SWAP) ≈ 1.0
    @test gate_concurrence(sqrt_SWAP') ≈ 1.0
    @test gate_concurrence(CPHASE8) ≈ abs(sin(0.5 * π / 8))
    @test gate_concurrence(SWAP) ≈ 0.0 atol = 1e-15

    @test gate_concurrence(0.2, 0.1, 0.1) < 1.0
    @test gate_concurrence(0.8, 0.1, 0.1) < 1.0
    @test gate_concurrence(0.5, 0.4, 0.1) ≈ 1.0
    @test gate_concurrence(0.5, 0.4, 0.2) < 1.0

end


@testset "PE Evaluation" begin

    @test D_PE(CNOT) ≈ 0.0
    @test D_PE(SWAP) ≈ -2.0
    @test D_PE(SWAP; absolute_square=true) ≈ 4.0
    @test D_PE(I(4)) ≈ 2.0
    @test D_PE(I(4); absolute_square=true) ≈ 4.0

end


@testset "Weyl Chamber Region" begin

    @test in_weyl_chamber(CNOT; region="W")
    @test in_weyl_chamber(CNOT; region="PE")
    @test weyl_chamber_region(CNOT) == "PE"
    @test !in_weyl_chamber(CNOT; region="W0")
    @test !in_weyl_chamber(CNOT; region="W0*")
    @test !in_weyl_chamber(CNOT; region="W1")

    @test in_weyl_chamber(I(4); region="W0")

    @test in_weyl_chamber(SWAP; region="W1")
    @test weyl_chamber_region(SWAP) == "W1"

    @test in_weyl_chamber(CPHASE8; region="W0")

    @test in_weyl_chamber(0.2, 0.1, 0.1)
    @test in_weyl_chamber(0.2, 0.1, 0.1; region="W0")
    @test !in_weyl_chamber(0.2, 0.1, 0.1; region="W1")

    @test in_weyl_chamber(0.8, 0.1, 0.1; region="W0*")

    @test in_weyl_chamber(0.5, 0.4, 0.4; region="W1")

    @test !in_weyl_chamber(1.1, 0.0, 0.0)
    @test !in_weyl_chamber(0.0, 0.6, 0.0)
    @test !in_weyl_chamber(0.0, 0.0, 0.6)
    @test !in_weyl_chamber(-0.1, 0.0, 0.0)
    @test !in_weyl_chamber(0.0, -0.1, 0.0)
    @test !in_weyl_chamber(0.0, 0.0, -0.1)
    @test !in_weyl_chamber(1.0, 0.1, 0.0)
    @test !in_weyl_chamber(0.6, 0.45, 0.0)
    @test !in_weyl_chamber(0.5, 0.0, 0.1)

    @test weyl_chamber_region(0.0, 0.0, -0.1) == ""

end


@testset "PE chi states" begin

    Ψ1 = random_state_vector(4)
    Ψ2 = random_state_vector(4)
    Ψ3 = random_state_vector(4)
    Ψ4 = random_state_vector(4)

    objectives = [
        Objective(
            initial_state=ket(lbl; N=2),
            generator=random_hermitian_real_matrix(4, 1.0)
        ) for lbl in ("00", "01", "10", "11")
    ]

    J_T = gate_functional(D_PE; unitarity_weight=0.0)
    chi_pe! = make_chi(J_T, objectives)
    ϕ = [Ψ1, Ψ2, Ψ3, Ψ4]
    χ_zygote = [similar(ϕₖ) for ϕₖ in ϕ]
    chi_pe!(χ_zygote, ϕ, objectives)

    gate_chi_pe! = make_gate_chi(D_PE, objectives; unitarity_weight=0.0)
    χ_gate_zygote = [similar(ϕₖ) for ϕₖ in ϕ]
    gate_chi_pe!(χ_gate_zygote, ϕ, objectives)

    gate_chi_pe_fd! =
        make_gate_chi(D_PE, objectives; use_finite_differences=true, unitarity_weight=0.0)
    χ_gate_fd = [similar(ϕₖ) for ϕₖ in ϕ]
    gate_chi_pe_fd!(χ_gate_fd, ϕ, objectives)

    vec_angle(v⃗, w⃗) = acos((v⃗ ⋅ w⃗) / (norm(v⃗) * norm(w⃗)))

    # Does Zygote gate gradient and FD gate gradient match?
    @test norm(χ_gate_zygote[1] - χ_gate_fd[1]) < 1e-8
    @test norm(χ_gate_zygote[2] - χ_gate_fd[2]) < 1e-8
    @test norm(χ_gate_zygote[3] - χ_gate_fd[3]) < 1e-8
    @test norm(χ_gate_zygote[4] - χ_gate_fd[4]) < 1e-8

    # Does the Zygote gate gradient and Zygote full gradient match?
    @test norm(χ_gate_zygote[1] - χ_zygote[1]) < 1e-8
    @test norm(χ_gate_zygote[2] - χ_zygote[2]) < 1e-8
    @test norm(χ_gate_zygote[3] - χ_zygote[3]) < 1e-8
    @test norm(χ_gate_zygote[4] - χ_zygote[4]) < 1e-8

    # Does the Zygote full gradient and the FD gate gradient match?
    @test norm(χ_zygote[1] - χ_gate_fd[1]) < 1e-8
    @test norm(χ_zygote[2] - χ_gate_fd[2]) < 1e-8
    @test norm(χ_zygote[3] - χ_gate_fd[3]) < 1e-8
    @test norm(χ_zygote[4] - χ_gate_fd[4]) < 1e-8

end


@testset "Concurrence-chi ($region)" for region in ("W0", "W0*", "W1")

    # Note: the χ states are known to fail for the PE region (but they'd be
    # zero anyway)

    J_T_val = 0.0
    local J_T_U
    local J_T
    local ϕ

    basis = [ket(lbl; N=2) for lbl in ("00", "01", "10", "11")]
    H = random_hermitian_real_matrix(4, 1.0)

    objectives = [Objective(initial_state=Ψ, generator=H) for Ψ in basis]

    c1, c2, c3 = rand(3) .* [1, 0.5, 0.5]
    while !in_weyl_chamber(c1, c2, c3; region)
        c1, c2, c3 = rand(3) .* [1, 0.5, 0.5]
    end

    U = canonical_gate(c1, c2, c3)
    ϕ = U' * basis

    J_T_U = U -> 1 - gate_concurrence(U)
    J_T = gate_functional(J_T_U)

    gate_chi_pe! = make_gate_chi(J_T_U, objectives)
    χ_gate_zygote = [similar(ϕₖ) for ϕₖ in ϕ]
    gate_chi_pe!(χ_gate_zygote, ϕ, objectives)

    gate_chi_pe_fd! = make_gate_chi(J_T_U, objectives; use_finite_differences=true)
    χ_gate_fd = [similar(ϕₖ) for ϕₖ in ϕ]
    gate_chi_pe_fd!(χ_gate_fd, ϕ, objectives)

    # Does Zygote gate gradient and FD gate gradient match?
    @test norm(χ_gate_zygote[1] - χ_gate_fd[1]) < 1e-8
    @test norm(χ_gate_zygote[2] - χ_gate_fd[2]) < 1e-8
    @test norm(χ_gate_zygote[3] - χ_gate_fd[3]) < 1e-8
    @test norm(χ_gate_zygote[4] - χ_gate_fd[4]) < 1e-8

end
