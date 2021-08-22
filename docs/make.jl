using QuantumControlBase
using Documenter

DocMeta.setdocmeta!(QuantumControlBase, :DocTestSetup, :(using QuantumControlBase); recursive=true)

makedocs(;
    modules=[QuantumControlBase],
    authors="Michael Goerz <mail@michaelgoerz.net>, Alastair Marshall <alastair@nvision-imaging.com>, and contributors",
    repo="https://github.com/quantumcontrol-jl/QuantumControlBase.jl/blob/{commit}{path}#{line}",
    sitename="QuantumControlBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://QuantumControl-jl.github.io/QuantumControlBase.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/QuantumControl-jl/QuantumControlBase.jl",
)
