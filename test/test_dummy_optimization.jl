using Test

using QuantumControlBase
using QuantumControlTestUtils.DummyOptimization: dummy_control_problem

@testset "dummy optimization" begin

    println("")
    problem = dummy_control_problem()
    result = optimize(problem; method=:dummymethod)
    @test result.converged

end
