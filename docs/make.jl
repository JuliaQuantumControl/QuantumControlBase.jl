using QuantumControlBase
using QuantumPropagators
using QuantumControl
using Documenter
using Pkg
using DocumenterCitations
using DocumenterInterLinks

DocMeta.setdocmeta!(
    QuantumControlBase,
    :DocTestSetup,
    :(using QuantumControlBase);
    recursive=true
)

links = InterLinks(
    "TimerOutputs" => (
        "https://github.com/KristofferC/TimerOutputs.jl",
        joinpath(@__DIR__, "src", "inventories", "TimerOutputs.toml")
    ),
    "QuantumPropagators" => "https://juliaquantumcontrol.github.io/QuantumPropagators.jl/dev/",
    "QuantumControl" => "https://juliaquantumcontrol.github.io/QuantumControl.jl/dev/",
)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
VERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/JuliaQuantumControl/QuantumControlBase.jl"

println("Starting makedocs")

makedocs(;
    plugins=[bib, links],
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
