# init file for "make devrepl"
using Revise
using JuliaFormatter
using LiveServer: serve, servedocs

println("""
*******************************************************************************
DEVELOPMENT REPL

Revise, JuliaFormatter, LiveServer are active.

* `include("test/runtests.jl")` – Run the entire test suite
* `include("docs/make.jl")` – Generate the documentation
* `format(".")` – Apply code formatting to all files
* `servedocs([port=8000, verbose=false])` –
  Build and serve the documentation. Automatically recompile and redisplay on
  changes
*******************************************************************************
""")
