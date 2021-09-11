using Test
using SafeTestsets

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose=true "QuantumControlBase" begin

    print("\n* Shapes (test_shapes.jl):")
    @time @safetestset "Shapes" begin include("test_shapes.jl") end

    print("\n* Discretization (test_discretization.jl):")
    @time @safetestset "Discretization" begin include("test_discretization.jl") end

    print("\n* Propagation (test_propagation.jl):")
    @time @safetestset "Propagation" begin include("test_propagation.jl") end

    print("\n* Liouvillian (test_liouvillian.jl):")
    @time @safetestset "Liouvillian" begin include("test_liouvillian.jl") end

    print("\n")

end
;
