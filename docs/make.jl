using QuantumControlBase
using QuantumPropagators
using QuantumControl
using Documenter
using Pkg
using DocumenterCitations
using DocumenterInterLinks

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/JuliaQuantumControl/QuantumControlBase.jl"

DEV_OR_STABLE = "stable/"
if endswith(VERSION, "dev")
    DEV_OR_STABLE = "dev/"
end


links = InterLinks(
    "TimerOutputs" => (
        "https://github.com/KristofferC/TimerOutputs.jl",
        joinpath(@__DIR__, "src", "inventories", "TimerOutputs.toml")
    ),
    "QuantumPropagators" => "https://juliaquantumcontrol.github.io/QuantumPropagators.jl/$DEV_OR_STABLE",
    "QuantumControl" => "https://juliaquantumcontrol.github.io/QuantumControl.jl/$DEV_OR_STABLE",
)

externals = ExternalFallbacks(
    "Generator" => "@extref QuantumPropagators :jl:type:`QuantumPropagators.Generators.Generator`",
    "QuantumPropagators.Interfaces.check_amplitude" => "@extref QuantumPropagators :jl:function:`QuantumPropagators.Interfaces.check_amplitude`",
    "evaluate" => "@extref QuantumPropagators :jl:function:`QuantumPropagators.Controls.evaluate`",
    "Operator" => "@extref QuantumPropagators :jl:type:`QuantumPropagators.Generators.Operator`",
    "QuantumPropagators.Interfaces.check_generator" => "@extref QuantumPropagators :jl:function:`QuantumPropagators.Interfaces.check_generator`",
    "QuantumPropagators.AbstractPropagator" => "@extref QuantumPropagators :jl:type:`QuantumPropagators.AbstractPropagator`",
    "QuantumPropagators.init_prop" => "@extref QuantumPropagators :jl:function:`QuantumPropagators.init_prop`",
    "QuantumPropagators.propagate" => "@extref QuantumPropagators :jl:function:`QuantumPropagators.propagate`",
    "QuantumControl.Functionals.J_T_sm" => "@extref QuantumControl :jl:function:`QuantumControl.Functionals.J_T_sm`",
    # "QuantumControl.Functionals.chi_sm" => "@extref QuantumControl :jl:function:`QuantumControl.Functionals.chi_sm`",
    "QuantumControl.Functionals.J_T_re" => "@extref QuantumControl :jl:function:`QuantumControl.Functionals.J_T_re`",
    # "QuantumControl.Functionals.chi_re" => "@extref QuantumControl :jl:function:`QuantumControl.Functionals.chi_re`",
    "QuantumControl.Functionals.J_T_ss" => "@extref QuantumControl :jl:function:`QuantumControl.Functionals.J_T_ss`",
    # "QuantumControl.Functionals.chi_ss" => "@extref QuantumControl :jl:function:`QuantumControl.Functionals.chi_ss`",
    "check_state" => "@extref QuantumPropagators :jl:function:`QuantumPropagators.Interfaces.check_state`"
)

println("Starting makedocs")

makedocs(;
    plugins=[bib, links, externals],
    authors=AUTHORS,
    sitename="QuantumControlBase.jl",
    warnonly=false,
    format=Documenter.HTML(;
        prettyurls=true,
        canonical="https://juliaquantumcontrol.github.io/QuantumControlBase.jl",
        assets=[
            asset(
                "https://juliaquantumcontrol.github.io/QuantumControl.jl/dev/assets/topbar/topbar.css"
            ),
            asset(
                "https://juliaquantumcontrol.github.io/QuantumControl.jl/dev/assets/topbar/topbar.js"
            ),
        ],
        footer="[$NAME.jl]($GITHUB) v$VERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)."
    ),
    pages=["Home" => "index.md"],
)

println("Finished makedocs")

deploydocs(; repo="github.com/JuliaQuantumControl/QuantumControlBase.jl")
