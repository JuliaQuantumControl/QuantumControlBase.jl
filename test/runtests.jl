using Test
using SafeTestsets

#=
# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose = true "QuantumControlBase" begin


    println("* Propagation (test_propagation.jl):")
    @time @safetestset "Propagation" begin
        include("test_propagation.jl")
    end

    println("* Derivatives (test_derives.jl):")
    @time @safetestset "Derivatives" begin
        include("test_derivs.jl")
    end

    println("\n* Parameterization (test_parameterization.jl):")
    @time @safetestset "Parameterization" begin
        include("test_parameterization.jl")
    end

    println("* Functionals (test_functionals.jl):")
    @time @safetestset "Functionals" begin
        include("test_functionals.jl")
    end

    println("* Callbacks (test_callbacks.jl):")
    @time @safetestset "Callbacks" begin
        include("test_callbacks.jl")
    end

    println("* Optimize-kwargs (test_optimize_kwargs.jl):")
    @time @safetestset "Optimize-kwargs" begin
        include("test_optimize_kwargs.jl")
    end

    println("* Dummy Optimization (test_dummy_optimization.jl):")
    @time @safetestset "Dummy Optimization" begin
        include("test_dummy_optimization.jl")
    end

    println("* Atexit dumps (test_atexit.jl):")
    @time @safetestset "Atexit dumps" begin
        include("test_atexit.jl")
    end

    println("* Trajectories (test_trajectories.jl):")
    @time @safetestset "Trajectories" begin
        include("test_trajectories.jl")
    end

    println("* Adjoint Trajectories (test_adjoint_trajectory.jl):")
    @time @safetestset "Adjoint Trajectories" begin
        include("test_adjoint_trajectory.jl")
    end

    println("* Control problems (test_control_problems.jl):")
    @time @safetestset "Control problems" begin
        include("test_control_problems.jl")
    end

    println("* Invalid interfaces (test_invalid_interfaces.jl):")
    @time @safetestset "Invalid interfaces" begin
        include("test_invalid_interfaces.jl")
    end

end;
nothing
=#
