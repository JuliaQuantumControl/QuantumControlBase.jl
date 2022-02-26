using Test
using SafeTestsets

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose = true "QuantumControlBase" begin

    print("\n* Shapes (test_shapes.jl):")
    @time @safetestset "Shapes" begin
        include("test_shapes.jl")
    end

    print("\n* Discretization (test_discretization.jl):")
    @time @safetestset "Discretization" begin
        include("test_discretization.jl")
    end

    print("\n* Controls (test_controls.jl):")
    @time @safetestset "Controls" begin
        include("test_controls.jl")
    end

    print("\n* Propagation (test_propagation.jl):")
    @time @safetestset "Propagation" begin
        include("test_propagation.jl")
    end

    print("\n* Liouvillian (test_liouvillian.jl):")
    @time @safetestset "Liouvillian" begin
        include("test_liouvillian.jl")
    end

    print("\n* GradGenerator (test_gradgen.jl):")
    @time @safetestset "GradGenerator" begin
        include("test_gradgen.jl")
    end

    print("\n* Infohook (test_infohook.jl):")
    @time @safetestset "Infohook" begin
        include("test_infohook.jl")
    end

    print("\n* Optimize-kwargs (test_optimize_kwargs.jl):")
    @time @safetestset "Optimize-kwargs" begin
        include("test_optimize_kwargs.jl")
    end

    print("\n* Dummy Optimization (test_dummy_optimization.jl):")
    @time @safetestset "Dummy Optimization" begin
        include("test_dummy_optimization.jl")
    end

    print("\n* Optimize-or-load (test_optimize_or_load.jl):")
    @time @safetestset "Optimize-or-load" begin
        include("test_optimize_or_load.jl")
    end

    print("\n")

end;
