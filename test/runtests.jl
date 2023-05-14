using Test
using SafeTestsets

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose = true "QuantumControlBase" begin


    print("\n* Propagation (test_propagation.jl):")
    @time @safetestset "Propagation" begin
        include("test_propagation.jl")
    end

    print("\n* Derivatives (test_derives.jl):")
    @time @safetestset "Derivatives" begin
        include("test_derivs.jl")
    end

    print("\n* Functionals (test_functionals.jl):")
    @time @safetestset "Functionals" begin
        include("test_functionals.jl")
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

    print("\n* Atexit dumps (test_atexit.jl):")
    @time @safetestset "Atexit dumps" begin
        include("test_atexit.jl")
    end

    print("\n* Adjoint Objectives (test_adjoint_objective.jl):")
    @time @safetestset "Adjoint Objectives" begin
        include("test_adjoint_objective.jl")
    end

    print("\n* Invalid interfaces (test_invalid_interfaces.jl):")
    @time @safetestset "Invalid interfaces" begin
        include("test_invalid_interfaces.jl")
    end

    print("\n")

end;
