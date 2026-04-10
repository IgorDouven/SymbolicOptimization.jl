using Test
using SymbolicOptimization
using Aqua

@testset "SymbolicOptimization.jl" begin
    @testset "Aqua quality checks" begin
        Aqua.test_all(SymbolicOptimization; ambiguities=false)
    end
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
