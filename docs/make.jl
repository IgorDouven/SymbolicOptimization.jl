using Documenter
using SymbolicOptimization

makedocs(
    sitename = "SymbolicOptimization.jl",
    modules = [SymbolicOptimization],
    authors = "Igor Douven",
    warnonly = [:missing_docs, :cross_references, :docs_block],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://IgorDouven.github.io/SymbolicOptimization.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "DSL Interface" => "dsl.md",
            "Model API" => "model.md",
            "Optimization" => "optimization.md",
        ],
        "Reference" => [
            "Core Types & Trees" => "types.md",
            "Grammar System" => "grammar.md",
            "Evaluation" => "evaluation.md",
            "Genetic Operators" => "operators.md",
        ],
        "Advanced" => "advanced.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/IgorDouven/SymbolicOptimization.jl",
    devbranch = "main",
    push_preview = true,
)
