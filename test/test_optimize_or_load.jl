using Test

using QuantumControlBase
using QuantumControlBase.TestUtils

@testset "dry-run" begin

    problem = dummy_control_problem()
    file_expected = joinpath(tempdir(), "dry-run.jld2")
    result, file =
        @test_logs (:info, "Would optimize and store in $file_expected") @optimize_or_load(
            tempdir(),
            problem;
            method=:dummymethod,
            filename="dry-run.jld2",
            dry_run=true,
            verbose=true
        )
    @test isnothing(result)
    @test file == file_expected

end


@testset "metadata" begin

    problem = dummy_control_problem()
    outdir = mktempdir()
    println("")
    result, file = @optimize_or_load(
        outdir,
        problem;
        method=:dummymethod,
        filename="optimization_with_metadata.jld2",
        metadata=Dict("testset" => "metadata", "method" => :dummymethod,)
    )
    @test result.converged
    @test basename(file) == "optimization_with_metadata.jld2"
    @test isfile(file)
    result_load, metadata = load_optimization(file; return_metadata=true)
    @test result_load isa QuantumControlBase.TestUtils.DummyOptimizationResult
    @test result_load.message == "Reached maximum number of iterations"
    @test metadata["testset"] == "metadata"
    @test metadata["method"] == :dummymethod

end


@testset "continue_from" begin
    problem = dummy_control_problem()
    outdir = mktempdir()
    println("")
    result, file = @optimize_or_load(
        outdir,
        problem;
        iter_stop=5,
        method=:dummymethod,
        filename="optimization_stage1.jld2"
    )
    @test result.converged
    result1 = load_optimization(file)
    result2 = optimize(problem; method=:dummymethod, iter_stop=15, continue_from=result1)
    @test result2.iter == 15
end
