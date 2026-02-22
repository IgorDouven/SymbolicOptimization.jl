using Test
using SymbolicOptimization

@testset "SymbolicOptimization.jl" begin
    include("test_types.jl")
    include("test_tree_utils.jl")
    include("test_printing.jl")
    include("test_grammar.jl")
    include("test_evaluation.jl")
    include("test_operators.jl")
    include("test_nsga2.jl")
    include("test_api.jl")
    include("test_symbolics.jl")
end
