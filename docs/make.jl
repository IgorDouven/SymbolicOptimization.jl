using Documenter
using SymbolicOptimization

makedocs(;
    modules = [SymbolicOptimization],
    sitename = "SymbolicOptimization.jl",
    authors = "Igor Douven",
    format = Documenter.HTML(;
        canonical = "https://IgorDouven.github.io/SymbolicOptimization.jl",
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/IgorDouven/SymbolicOptimization.jl",
    devbranch = "main",
)
