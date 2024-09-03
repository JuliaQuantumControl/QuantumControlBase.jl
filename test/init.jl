# init file for "make devrepl"
using Revise
using JuliaFormatter
# using QuantumControlTestUtils: test, show_coverage, generate_coverage_html
import Pkg
using LiveServer: LiveServer, serve, servedocs
include(joinpath(@__DIR__, "clean.jl"))


function test(
    file="test/runtests.jl";
    root=pwd(),
    project="test",
    code_coverage=joinpath(".coverage", "tracefile-%p.info"),
    color=(Base.have_color === nothing ? "auto" : Base.have_color ? "yes" : "no"),
    compiled_modules=(Bool(Base.JLOptions().use_compiled_modules) ? "yes" : "no"),
    startup_file=(Base.JLOptions().startupfile == 1 ? "yes" : "no"),
    depwarn=(Base.JLOptions().depwarn == 2 ? "error" : "yes"),
    inline=(Bool(Base.JLOptions().can_inline) ? "yes" : "no"),
    track_allocation=(("none", "user", "all")[Base.JLOptions().malloc_log+1]),
    check_bounds="yes",
    threads=Threads.nthreads(),
)
    root = abspath(normpath(root))
    if !(startswith(code_coverage, "@") || code_coverage in ["none", "user", "all"])
        code_coverage = normpath(root, code_coverage)
        if !endswith(code_coverage, ".info")
            @error "`code_coverage` must have `.info` extension" code_coverage
        end
        mkpath(dirname(code_coverage))
    end
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
end


REPL_MESSAGE = """
*******************************************************************************
DEVELOPMENT REPL

Revise, JuliaFormatter, LiveServer are active.

* `help()` – Show this message
* `include("test/runtests.jl")` – Run the entire test suite
* `test()` – Run the entire test suite in a subprocess with coverage
* `show_coverage()` – Print a tabular overview of coverage data
* `generate_coverage_html()` – Generate an HTML coverage report
* `Pkg.test("Krotov", coverage=true)` –
  Run upstream Krotov tests for additional coverage
* `include("docs/make.jl")` – Generate the documentation
* `format(".")` – Apply code formatting to all files
* `servedocs([port=8000, verbose=false])` –
  Build and serve the documentation. Automatically recompile and redisplay on
  changes
* `clean()` – Clean up build/doc/testing artifacts
* `distclean()` – Restore to a clean checkout state
*******************************************************************************
"""

"""Show help"""
help() = println(REPL_MESSAGE)
