using Test
import QuantumControlBase
using QuantumControl: ControlProblem, Objective

@testset "optimize-kwargs" begin

    # test that we can call optimize with kwargs that override the kwargs of
    # the `problem` without permantly changing `problem`


    struct Result
        iter_stop::Int
        flag::Bool
    end

    function optimize_kwargstest(problem)
        return Result(problem.kwargs[:iter_stop], problem.kwargs[:flag])
    end

    QuantumControlBase.optimize(problem, method::Val{:kwargstest}) =
        optimize_kwargstest(problem)

    problem = ControlProblem(
        objectives=[Objective(initial_state=nothing, generator=nothing)],
        pulse_options=Dict(),
        tlist=[0.0, 10.0],
        iter_stop=2,
        flag=false
    )

    res = QuantumControlBase.optimize(problem; method=:kwargstest, check=false)
    @test res.iter_stop == 2
    @test !res.flag

    res2 = QuantumControlBase.optimize(
        problem;
        method=:kwargstest,
        iter_stop=10,
        flag=true,
        check=false
    )
    @test res2.iter_stop == 10
    @test res2.flag
    @test problem.kwargs[:iter_stop] == 2
    @test !problem.kwargs[:flag]

end
