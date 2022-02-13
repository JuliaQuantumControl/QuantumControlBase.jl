module TestUtils

export random_complex_matrix, random_real_matrix, random_hermitian_matrix
export random_complex_sparse_matrix, random_real_sparse_matrix
export random_hermitian_sparse_matrix, random_state_vector
export test

using Logging
using Random
using Distributions
using LinearAlgebra
using SparseArrays
using Coverage
using LocalCoverage


"""Run a package test-suite in a subprocess.

```julia
test(
    file="test/runtests.jl";
    root=pwd(),
    project="test",
    code_coverage="user",
    show_coverage=(code_coverage == "user"),
    color="auto",
    startup_file="yes",
    depwarn="yes",
    check_bounds="yes",
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
in `covdir`. This requires the `genhtml` executable (part of the
[lcov](http://ltp.sourceforge.net/coverage/lcov.php) package). Instead of
`true`, it is also possible to pass the path to the `genhtml` exectuable.

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
    color="auto",
    startup_file="yes",
    depwarn="yes",
    check_bounds="yes",
    genhtml::Union{Bool,AbstractString}=false,
    covdir="coverage"
)
    julia = Base.julia_cmd().exec[1]
    cmd = [
        julia,
        "--project=$project",
        "--color=$color",
        "--startup-file=$startup_file",
        "--code-coverage=$code_coverage",
        "--depwarn=$depwarn",
        "--check-bounds=$check_bounds",
        "-e",
        "include(\"$file\")"
    ]
    @info "Running '$(join(cmd, " "))' in subprocess"
    run(Cmd(Cmd(cmd), dir=root))
    tracefile = joinpath(root, "lcov.info")
    if show_coverage || genhtml
        logger = Logging.SimpleLogger(Logging.Error)
        local coverage
        Logging.with_logger(logger) do
            coverage = Coverage.process_folder(joinpath(root, "src"))
        end
        if show_coverage
            coverage_summary(coverage)
        end
        (genhtml === true) && (genhtml = "genhtml")
        (genhtml === false) && (genhtml = "")
        if !isempty(genhtml)
            LocalCoverage.CoverageTools.LCOV.writefile(
                joinpath(covdir, tracefile),
                coverage
            )
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
            @info("Generated coverage HTML. Serve with 'LiveServer.serve(dir=\"$covdir\")'")
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

end
