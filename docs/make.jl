using QuantumControlBase
using QuantumPropagators
using QuantumControl
using Documenter
using Pkg
using DocumenterInterLinks

DocMeta.setdocmeta!(
    QuantumControlBase,
    :DocTestSetup,
    :(using QuantumControlBase);
    recursive=true
)

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/JuliaQuantumControl/QuantumControlBase.jl"

println("Starting makedocs")

makedocs(;
    authors=AUTHORS,
    sitename="QuantumControlBase.jl",
    warnonly=true,
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
