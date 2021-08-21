using Test
using SafeTestsets

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose=true "QuantumControlBase" begin

    print("\n* Shapes (test_shapes.jl):")
    @time @safetestset "Shapes" begin include("test_shapes.jl") end

    print("\n")

end
;
