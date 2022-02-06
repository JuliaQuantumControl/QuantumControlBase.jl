using QuantumControlBase
using QuantumPropagators
using Documenter
using Pkg

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
    modules=[QuantumControlBase],
    format=Documenter.HTML(;
        prettyurls=true,
        canonical="https://juliaquantumcontrol.github.io/QuantumControlBase.jl",
        assets=String[],
        footer="[$NAME.jl]($GITHUB) v$VERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)."
    ),
    pages=[
        "Home" => "index.md",
        hide("QuantumPropagators" => "quantumpropagators.md"),
        "History" => "history.md",
    ]
)

println("Finished makedocs")

deploydocs(; repo="github.com/JuliaQuantumControl/QuantumControlBase.jl")
