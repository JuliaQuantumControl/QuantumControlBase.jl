module TestUtils

export random_complex_matrix,
    random_real_matrix,
    random_hermitian_matrix,
    random_complex_sparse_matrix,
    random_real_sparse_matrix,
    random_hermitian_sparse_matrix,
    random_state_vector,
    random_hermitian_real_matrix,
    random_hermitian_sparse_real_matrix
export dummy_control_problem
export test
export show_coverage, generate_coverage_html

using Logging
using Random
using Distributions
using LinearAlgebra
using SparseArrays
using Coverage
using PrettyTables
using LocalCoverage:
    LocalCoverage,
    eval_coverage_metrics,
    PackageCoverage,
    FileCoverageSummary,
    CoverageTools
using Printf
using QuantumPropagators.Controls: getcontrols, discretize, discretize_on_midpoints

import ..Objective
import ..ControlProblem


function format_line(metric::Union{PackageCoverage,FileCoverageSummary})
    name = (metric isa PackageCoverage ? "TOTAL" : relpath(metric.filename))
    lines_hit = @sprintf("%3d", metric.lines_hit)
    lines_tracked = @sprintf("%3d", metric.lines_tracked)
    lines_missing = @sprintf("%3d", metric.lines_tracked - metric.lines_hit)
    coverage = (isnan(metric.coverage) ? "-" : @sprintf("%3.0f%%", metric.coverage))
    return hcat(name, lines_tracked, lines_hit, lines_missing, coverage)
end


"""Print out a coverage summary from existing coverage data.

```julia
show_coverage(path="./src"; sort_by=nothing)
```

prints a a table showing the tracked files in `path`, the total number of
tracked lines in that file ("Total"), the number of lines with coverage
("Hit"), the number of lines without coverage ("Missed") and the "Coverage" as
a percentage.

The coverage data is collected from `.cov` files in `path`.

Optionally, the table can be sorted by passing the name of a column to
`sort_by`, e..g. `sort_py=:Missed`.
"""
function show_coverage(path::String=joinpath(pwd(), "src"); kwargs...)
    local coverage
    logger = Logging.SimpleLogger(Logging.Error)
    Logging.with_logger(logger) do
        coverage = Coverage.process_folder(path)
    end
    metrics = eval_coverage_metrics(coverage, path)
    show_coverage(metrics; kwargs...)
end

function show_coverage(metrics::PackageCoverage; sort_by=nothing)

    file_metrics = metrics.files
    sorter = Dict(
        :Total => (m -> m.lines_tracked),
        :Hit => (m -> m.lines_hit),
        :Missed => (m -> (m.lines_tracked - m.lines_hit)),
        :Coverage => (m -> m.coverage),
    )
    if !isnothing(sort_by)
        if sort_by ∈ keys(sorter)
            sort!(file_metrics; by=sorter[sort_by])
        else
            error("Cannot sort by $sort_by, must be one of $(keys(sorter))")
        end
    end

    table = reduce(vcat, map(format_line, [file_metrics..., metrics]))

    row_coverage = [getfield.(file_metrics, :coverage)... metrics.coverage]

    highlighters = (
        Highlighter(
            (data, i, j) -> j == 5 && row_coverage[i] <= 50,
            bold=true,
            foreground=:red,
        ),
        Highlighter((data, i, j) -> j == 5 && row_coverage[i] >= 90, foreground=:green),
    )

    table_str = pretty_table(
        table,
        header=["File name", "Total", "Hit", "Missed", "Coverage"],
        alignment=[:l, :r, :r, :r, :r],
        crop=:none,
        linebreaks=true,
        columns_width=[min(30, maximum(length.(table[:, 1]))), 6, 6, 6, 8],
        autowrap=false,
        highlighters=highlighters,
        body_hlines=[size(table, 1) - 1],
    )
    println(table_str)

end

_show_coverage_func = show_coverage


"""Generate an HTML report for existing coverage data.

```julia
generate_coverage_html(path="./"; covdir="coverage", genhtml="genhtml")
```

creates a folder `covdir` and use the external `genhtml` program to write an
HTML coverage report into that folder.
"""
function generate_coverage_html(path::String=pwd(); kwargs...)
    local coverage
    logger = Logging.SimpleLogger(Logging.Error)
    Logging.with_logger(logger) do
        coverage = Coverage.process_folder(path)
    end
    generate_coverage_html(path, coverage; kwargs...)
end

function generate_coverage_html(
    path::String,
    coverage::Vector{LocalCoverage.CoverageTools.FileCoverage};
    covdir="coverage",
    genhtml="genhtml"
)
    covdir = joinpath(path, covdir)
    tracefile = joinpath(path, "lcov.info")
    LocalCoverage.CoverageTools.LCOV.writefile(tracefile, coverage)
    branch = try
        strip(read(`git rev-parse --abbrev-ref HEAD`, String))
    catch e
        @warn "git branch could not be detected.\nError message: $(sprint(Base.showerror, e))"
    end
    title = isnothing(branch) ? "N/A" : "on branch $(branch)"
    try
        run(`$(genhtml) -t $(title) -o $(covdir) $(tracefile)`)
    catch e
        @error(
            "Failed to run $(genhtml). Check that lcov is installed.\nError message: $(sprint(Base.showerror, e))"
        )
    end
    @info(
        "Generated coverage HTML. Serve with 'LiveServer.serve(dir=\"$(relpath(covdir, pwd()))\")'"
    )
end


"""Run a package test-suite in a subprocess.

```julia
test(
    file="test/runtests.jl";
    root=pwd(),
    project="test",
    code_coverage="user",
    show_coverage=(code_coverage == "user"),
    color=<inherit>,
    compiled_modules=<inherit>,
    startup_file=<inherit>,
    depwarn=<inherit>,
    inline=<inherit>,
    check_bounds="yes",
    track_allocation=<inherit>,
    threads=<inherit>,
    genhtml=false,
    covdir="coverage"
)
```

runs the test suite of the package located at `root` by running `include(file)`
inside a new julia process.

This is similar to what `Pkg.test()` does, but differs in the "sandboxing"
approach. While `Pkg.test()` creates a new temporary sandboxed environment,
`test()` uses an existing environment in `project` (the `test` subfolder by
default). This allows testing against the dev-versions of other packages. It
requires that the `test` folder contains both a `Project.toml` and a
`Manifest.toml` file.

The `test()` function also differs from directly including `test/runtests.jl`
in the REPL in that it can generate coverage data and reports (this is only
possible when running tests in a subprocess).

If `show_coverage` is passed as `true` (default), a coverage summary is shown.
Further, if `genhtml` is `true`, a full HTML coverage report will be generated
in `covdir` (relative to `root`). This requires the `genhtml` executable (part
of the [lcov](http://ltp.sourceforge.net/coverage/lcov.php) package). Instead
of `true`, it is also possible to pass the path to the `genhtml` exectuable.

All other keyword arguments correspond to the respective command line flag for
the `julia` executable that is run as the subprocess.

This function is intended to be exposed in a project's development-REPL.
"""
function test(
    file="test/runtests.jl";
    root=pwd(),
    project="test",
    code_coverage="user",
    show_coverage=(code_coverage == "user"),
    color=(Base.have_color === nothing ? "auto" : Base.have_color ? "yes" : "no"),
    compiled_modules=(Bool(Base.JLOptions().use_compiled_modules) ? "yes" : "no"),
    startup_file=(Base.JLOptions().startupfile == 1 ? "yes" : "no"),
    depwarn=(Base.JLOptions().depwarn == 2 ? "error" : "yes"),
    inline=(Bool(Base.JLOptions().can_inline) ? "yes" : "no"),
    track_allocation=(("none", "user", "all")[Base.JLOptions().malloc_log+1]),
    check_bounds="yes",
    threads=Threads.nthreads(),
    genhtml::Union{Bool,AbstractString}=false,
    covdir="coverage"
)
    julia = Base.julia_cmd().exec[1]
    cmd = [
        julia,
        "--project=$project",
        "--color=$color",
        "--compiled-modules=$compiled_modules",
        "--startup-file=$startup_file",
        "--code-coverage=$code_coverage",
        "--track-allocation=$track_allocation",
        "--depwarn=$depwarn",
        "--check-bounds=$check_bounds",
        "--threads=$threads",
        "--inline=$inline",
        "--eval",
        "include(\"$file\")"
    ]
    @info "Running '$(join(cmd, " "))' in subprocess"
    run(Cmd(Cmd(cmd), dir=root))
    if show_coverage || genhtml
        logger = Logging.SimpleLogger(Logging.Error)
        local coverage
        package_dir = joinpath(root, "src")
        Logging.with_logger(logger) do
            coverage = Coverage.process_folder(package_dir)
        end
        if show_coverage
            metrics = eval_coverage_metrics(coverage, package_dir)
            _show_coverage_func(metrics)
        end
        (genhtml === true) && (genhtml = "genhtml")
        (genhtml === false) && (genhtml = "")
        if !isempty(genhtml)
            generate_html_coverage(root; covdir, genhtml)
        end
    end
end




"""Construct a random complex matrix of size N×N with spectral radius ρ.

```julia
random_complex_matrix(N, ρ)
```
"""
function random_complex_matrix(N, ρ)
    σ = 1 / √N
    d = Normal(0.0, σ)
    H = ρ * (rand(d, (N, N)) + rand(d, (N, N)) * 1im) / √2
end


"""Construct a random real-valued matrix of size N×N with spectral radius ρ.

```julia
random_real_matrix(N, ρ)
```
"""
function random_real_matrix(N, ρ)
    σ = 1 / √N
    d = Normal(0.0, σ)
    H = ρ * rand(d, (N, N))
end


"""Construct a random Hermitian matrix of size N×N with spectral radius ρ.

```julia
random_hermitian_matrix(N, ρ)
```
"""
function random_hermitian_matrix(N, ρ)
    σ = 1 / √N
    d = Normal(0.0, σ)
    X = (rand(d, (N, N)) + rand(d, (N, N)) * 1im) / √2
    H = ρ * (X + X') / (2 * √2)
end


"""Construct a random Hermitian real matrix of size N×N with spectral radius ρ.

```julia
random_hermitian_real_matrix(N, ρ)
```
"""
function random_hermitian_real_matrix(N, ρ)
    σ = 1 / √N
    d = Normal(0.0, σ)
    X = rand(d, (N, N))
    H = ρ * (X + X') / (2 * √2)
end


"""Construct a random sparse complex matrix.

```julia
random_complex_sparse_matrix(N, ρ, sparsity)
```

returns a matrix of size N×N with spectral radius ρ and the given sparsity
(number between zero and one that is the approximate fraction of non-zero
elements).
"""
function random_complex_sparse_matrix(N, ρ, sparsity)
    σ = 1 / √(sparsity * N)
    d = Normal(0.0, σ)
    Hre = sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
    Him = sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
    H = ρ * (Hre + Him * 1im) / √2
end


"""Construct a random sparse real-valued matrix.

```julia
random_real_sparse_matrix(N, ρ, sparsity)
```

returns a matrix of size N×N with spectral radius ρ and the given sparsity
(number between zero and one that is the approximate fraction of non-zero
elements).
"""
function random_real_sparse_matrix(N, ρ, sparsity)
    σ = 1 / √(sparsity * N)
    d = Normal(0.0, σ)
    H = ρ * sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
end


"""Construct a random sparse Hermitian matrix.

```julia
random_hermitian_sparse_matrix(N, ρ, sparsity)
```

returns a matrix of size N×N with spectral radius ρ and the given sparsity
(number between zero and one that is the approximate fraction of non-zero
elements).
"""
function random_hermitian_sparse_matrix(N, ρ, sparsity)
    σ = 1 / √(sparsity * N)
    d = Normal(0.0, σ)
    H1 = sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
    H2 = copy(H1)
    H2.nzval .= rand(d, length(H2.nzval))
    X = (H1 + H2 * 1im) / √2
    return 0.5ρ * (X + X') / √2
end


"""Construct a random sparse Hermitian real matrix.

```julia
random_hermitian_sparse_real_matrix(N, ρ, sparsity)
```

returns a matrix of size N×N with spectral radius ρ and the given sparsity
(number between zero and one that is the approximate fraction of non-zero
elements).
"""
function random_hermitian_sparse_real_matrix(N, ρ, sparsity)
    σ = 1 / √(sparsity * N)
    d = Normal(0.0, σ)
    H = sprand(N, N, sparsity, (dims...) -> rand(d, dims...))
    return 0.5ρ * (H + H') / √2
end


"""Return a random, normalized Hilbert space state vector of dimension `N`.

```julia
random_state_vector(N)
```
"""
function random_state_vector(N)
    Ψ = rand(N) .* exp.((2π * im) .* rand(N))
    Ψ ./= norm(Ψ)
    return Ψ
end


"""Set up a dummy control problem.

```julia
problem = dummy_control_problem(;
    N=10, n_objectives=1, n_controls=1, n_steps=50, dt=1.0, sparsity=0.5,
    complex_operators=true, hermitian=true, kwargs...)
```

Sets up a control problem with random (sparse) Hermitian matrices.

# Arguments

* `N`: The dimension of the Hilbert space
* `n_objectives`: The number of objectives in the optimization. All objectives
  will have the same Hamiltonian, but random initial and target states.
* `n_controls`: The number of controls, that is, the number of control terms in
  the control Hamiltonian. Each control is an array of random values,
  normalized on the intervals of the time grid.
* `n_steps`: The number of time steps (intervals of the time grid)
* `dt`: The time step
* `sparsity`: The sparsity of the Hamiltonians, as a number between 0.0 and
  1.0. For `sparsity=1.0`, the Hamiltonians will be dense matrices.
* `complex_operators`: Whether or not the drift/control operators will be
  complex-valued or real-valued.
* `hermitian`: Whether or not all drift/control operators will be Hermitian matrices.
* `kwargs`: All other keyword arguments are passed on to
  [`ControlProblem`](@ref)
"""
function dummy_control_problem(;
    N=10,
    n_objectives=1,
    n_controls=1,
    n_steps=50,
    dt=1.0,
    sparsity=0.5,
    complex_operators=true,
    hermitian=true,
    kwargs...
)

    tlist = collect(range(0; length=(n_steps + 1), step=dt))
    pulses = [rand(length(tlist) - 1) for l = 1:n_controls]
    for l = 1:n_controls
        # we normalize on the *intervals*, not on the time grid points
        pulses[l] ./= norm(pulses[l])
    end
    controls = [discretize(pulse, tlist) for pulse in pulses]

    function random_op(N, ρ, sparsity, complex_operators, hermitian)
        if sparsity < 1.0
            if hermitian
                if complex_operators
                    H = random_hermitian_sparse_matrix(N, ρ, sparsity)
                end
            else
                if complex_operators
                    H = random_complex_sparse_matrix(N, ρ, sparsity)
                end
            end
        else
            if hermitian
                if complex_operators
                    H = random_hermitian_matrix(N, ρ)
                else
                    H = random_hermitian_real_matrix(N, ρ)
                end
            else
                if complex_operators
                    H = random_complex_matrix(N, ρ)
                else
                    H = random_real_matrix(N, ρ)
                end
            end
        end
    end

    hamiltonian = []
    H_0 = random_op(N, 1.0, sparsity, complex_operators, hermitian)
    push!(hamiltonian, H_0)
    for control ∈ controls
        H_c = random_op(N, 1.0, sparsity, complex_operators, hermitian)
        push!(hamiltonian, (H_c, control))
    end

    objectives = [
        Objective(;
            initial_state=random_state_vector(N),
            generator=tuple(hamiltonian...),
            target_state=random_state_vector(N)
        ) for k = 1:n_objectives
    ]

    return ControlProblem(
        objectives=objectives,
        pulse_options=Dict(
            control => Dict(:lambda_a => 1.0, :update_shape => t -> 1.0) for
            control in controls
        ),
        tlist=tlist,
        kwargs...
    )
end


"""Result returned by [`optimize_with_dummy_method`](@ref)."""
mutable struct DummyOptimizationResult
    tlist::Vector{Float64}
    iter_start::Int64  # the starting iteration number
    iter_stop::Int64 # the maximum iteration number
    iter::Int64  # the current iteration number
    J_T::Float64  # the current value of the final-time functional J_T
    J_T_prev::Float64  # previous value of J_T
    guess_controls::Vector{Vector{Float64}}
    optimized_controls::Vector{Vector{Float64}}
    converged::Bool
    message::String

    function DummyOptimizationResult(problem)
        tlist = problem.tlist
        controls = getcontrols(problem.objectives)
        iter_start = get(problem.kwargs, :iter_start, 0)
        iter = iter_start
        iter_stop = get(problem.kwargs, :iter_stop, 20)
        guess_controls = [discretize(control, tlist) for control in controls]
        J_T = 0.0
        J_T_prev = 0.0
        optimized_controls = [copy(guess) for guess in guess_controls]
        converged = false
        message = "in progress"
        new(
            tlist,
            iter_start,
            iter_stop,
            iter,
            J_T,
            J_T_prev,
            guess_controls,
            optimized_controls,
            converged,
            message
        )
    end

end

struct DummyOptimizationWrk
    objectives
    adjoint_objectives
    kwargs
    controls
    pulses0::Vector{Vector{Float64}}
    pulses1::Vector{Vector{Float64}}
    result
end


function DummyOptimizationWrk(problem)
    objectives = [obj for obj in problem.objectives]
    adjoint_objectives = [adjoint(obj) for obj in problem.objectives]
    controls = getcontrols(objectives)
    kwargs = Dict(problem.kwargs)
    tlist = problem.tlist
    if haskey(kwargs, :continue_from)
        @info "Continuing previous optimization"
        result = kwargs[:continue_from]
        if !(result isa DummyOptimizationResult)
            result = convert(DummyOptimizationResult, result)
        end
        result.iter_stop = get(problem.kwargs, :iter_stop, 20)
        result.converged = false
        result.message = "in progress"
        pulses0 = [
            discretize_on_midpoints(control, tlist) for control in result.optimized_controls
        ]
    else
        result = DummyOptimizationResult(problem)
        pulses0 = [discretize_on_midpoints(control, tlist) for control in controls]
    end
    pulses1 = [copy(pulse) for pulse in pulses0]
    return DummyOptimizationWrk(
        objectives,
        adjoint_objectives,
        kwargs,
        controls,
        pulses0,
        pulses1,
        result,
    )
end

function update_result!(wrk::DummyOptimizationWrk, ϵ⁽ⁱ⁺¹⁾, i::Int64)
    res = wrk.result
    res.J_T_prev = res.J_T
    res.J_T = sum([norm(ϵ) for ϵ ∈ ϵ⁽ⁱ⁺¹⁾])
    (i > 0) && (res.iter = i)
    if i >= res.iter_stop
        res.converged = true
        res.message = "Reached maximum number of iterations"
    end
end


"""Run a dummy optimization.

```julia
result = optimize(problem, method=:dummymethod)
```

runs through and "optimization" of the given `problem` where in each iteration,
the amplitude of the guess pulses is diminished by 10%. The (summed) vector
norm of the the control serves as the value of the optimization functional.
"""
function optimize_with_dummy_method(problem)
    # This is connected to the main `optimize` method in the main
    # QuantumcontrolBase.jl
    iter_start = get(problem.kwargs, :iter_start, 0)
    check_convergence! = get(problem.kwargs, :check_convergence, res -> res)
    wrk = DummyOptimizationWrk(problem)
    ϵ⁽ⁱ⁾ = wrk.pulses0
    ϵ⁽ⁱ⁺¹⁾ = wrk.pulses1
    update_result!(wrk, ϵ⁽ⁱ⁺¹⁾, 0)
    println("# iter\tJ_T")
    @printf("%6d\t%.2e\n", 0, wrk.result.J_T)
    i = wrk.result.iter  # = 0, unless continuing from previous optimization
    while !wrk.result.converged
        i = i + 1
        for l = 1:length(ϵ⁽ⁱ⁺¹⁾)
            ϵ⁽ⁱ⁺¹⁾[l] .= 0.9 * ϵ⁽ⁱ⁾[l]
        end
        update_result!(wrk, ϵ⁽ⁱ⁺¹⁾, i)
        @printf("%6d\t%.2e\n", i, wrk.result.J_T)
        check_convergence!(wrk.result)
        ϵ⁽ⁱ⁾, ϵ⁽ⁱ⁺¹⁾ = ϵ⁽ⁱ⁺¹⁾, ϵ⁽ⁱ⁾
    end
    for l = 1:length(ϵ⁽ⁱ⁾)
        wrk.result.optimized_controls[l] = discretize(ϵ⁽ⁱ⁾[l], problem.tlist)
    end
    return wrk.result
end


end
