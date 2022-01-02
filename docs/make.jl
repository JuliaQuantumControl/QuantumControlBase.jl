using QuantumControlBase
using QuantumPropagators
using Documenter

DocMeta.setdocmeta!(QuantumControlBase, :DocTestSetup, :(using QuantumControlBase); recursive=true)

makedocs(;
    modules=[QuantumControlBase],
    authors="Michael Goerz <mail@michaelgoerz.net>, Alastair Marshall <alastair@nvision-imaging.com>, and contributors",
    sitename="QuantumControlBase.jl",
    format=Documenter.HTML(;
        prettyurls=true,
        canonical="https://juliaquantumcontrol.github.io/QuantumControlBase.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        hide("QuantumPropagators" => "quantumpropagators.md"),
        "History" => "history.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaQuantumControl/QuantumControlBase.jl",
)
