using Symbolics

# Alias to disambiguate from Symbolics.Variable / DynamicPolynomials.Variable
const SOVariable = SymbolicOptimization.Variable
const SOConstant = SymbolicOptimization.Constant
const SOFunctionNode = SymbolicOptimization.FunctionNode

@testset "Symbolics.jl Integration" begin
    
    @testset "to_symbolics" begin
        # Simple variable
        tree = SOVariable(:x)
        sym = to_symbolics(tree)
        @test string(sym) == "x"
        
        # Constant
        tree = SOConstant(3.14)
        sym = to_symbolics(tree)
        @test sym ≈ 3.14
        
        # x + 1
        tree = SOFunctionNode(:+, SOVariable(:x), SOConstant(1.0))
        sym = to_symbolics(tree)
        s = string(sym)
        @test contains(s, "x") && contains(s, "1")
        
        # sin(x)
        tree = SOFunctionNode(:sin, SOVariable(:x))
        sym = to_symbolics(tree)
        @test string(sym) == "sin(x)"
        
        # Safe operators map to standard ones
        tree = SOFunctionNode(:safe_div, SOVariable(:x), SOVariable(:y))
        sym = to_symbolics(tree)
        @test contains(string(sym), "/") || contains(string(sym), "x")
        
        tree = SOFunctionNode(:safe_log, SOVariable(:x))
        sym = to_symbolics(tree)
        @test string(sym) == "log(x)"
    end
    
    @testset "from_symbolics" begin
        Symbolics.@variables x y
        
        # Simple variable
        tree = from_symbolics(x)
        @test tree isa SOVariable
        @test tree.name == :x
        
        # Constant
        tree = from_symbolics(2.0)
        @test tree isa SOConstant
        @test tree.value == 2.0
        
        # Addition
        tree = from_symbolics(x + y)
        @test tree isa SOFunctionNode
        @test tree.func == :+
        
        # Nested: sin(x)
        tree = from_symbolics(sin(x))
        @test tree isa SOFunctionNode
        @test tree.func == :sin
        @test tree.children[1] isa SOVariable
    end
    
    @testset "Round-trip conversion" begin
        # x + 1
        orig = SOFunctionNode(:+, SOVariable(:x), SOConstant(1.0))
        sym = to_symbolics(orig)
        back = from_symbolics(sym)
        @test back isa SOFunctionNode || back isa SOVariable
        
        # sin(x * y)
        orig = SOFunctionNode(:sin, SOFunctionNode(:*, SOVariable(:x), SOVariable(:y)))
        sym = to_symbolics(orig)
        back = from_symbolics(sym)
        @test back isa SOFunctionNode
    end
    
    @testset "deep_simplify" begin
        # x + 0 → x
        tree = SOFunctionNode(:+, SOVariable(:x), SOConstant(0.0))
        s = deep_simplify(tree)
        @test s isa SOVariable
        @test s.name == :x
        
        # x * 1 → x
        tree = SOFunctionNode(:*, SOVariable(:x), SOConstant(1.0))
        s = deep_simplify(tree)
        @test s isa SOVariable
        @test s.name == :x
        
        # 0 * x → 0
        tree = SOFunctionNode(:*, SOConstant(0.0), SOVariable(:x))
        s = deep_simplify(tree)
        @test s isa SOConstant
        @test s.value == 0.0
        
        # (x + 1) - 1 → x
        tree = SOFunctionNode(:-, 
            SOFunctionNode(:+, SOVariable(:x), SOConstant(1.0)),
            SOConstant(1.0))
        s = deep_simplify(tree)
        @test s isa SOVariable
        @test s.name == :x
        
        # x * x → x^2
        tree = SOFunctionNode(:*, SOVariable(:x), SOVariable(:x))
        s = deep_simplify(tree)
        str = node_to_string(s)
        @test contains(str, "x") 
        
        # Safe operators: safe_div(x, 1) → x
        tree = SOFunctionNode(:safe_div, SOVariable(:x), SOConstant(1.0))
        s = deep_simplify(tree)
        @test s isa SOVariable
        @test s.name == :x
        
        # Constant folding: 2 + 3 → 5
        tree = SOFunctionNode(:+, SOConstant(2.0), SOConstant(3.0))
        s = deep_simplify(tree)
        @test s isa SOConstant
        @test s.value ≈ 5.0
    end
    
    @testset "simplified_string" begin
        tree = SOFunctionNode(:+, SOVariable(:x), SOConstant(0.0))
        s = simplified_string(tree)
        @test s == "x"
        
        tree = SOFunctionNode(:*, SOConstant(2.0), SOVariable(:x))
        s = simplified_string(tree)
        @test contains(s, "x") && contains(s, "2")
    end
    
    @testset "simplified_latex" begin
        tree = SOFunctionNode(:/, SOVariable(:x), SOVariable(:y))
        s = simplified_latex(tree)
        @test contains(s, "frac") || contains(s, "/")
        @test contains(s, "x") && contains(s, "y")
    end
    
    @testset "simplify_piecewise" begin
        # Build: step_func(x - y) * (x / z) + (1 - step_func(x - y)) * (y / z)
        # If-branch (step=1): x/z
        # Else-branch (step=0): y/z  (after simplifying 1*y/z and 0*x/z)
        cond = SOFunctionNode(:-, SOVariable(:x), SOVariable(:y))
        step_node = SOFunctionNode(:step_func, cond)
        one = SOFunctionNode(:/, SOVariable(:x), SOVariable(:x))  # x/x = 1
        one_minus_step = SOFunctionNode(:-, copy_tree(one), copy_tree(step_node))
        
        if_part = SOFunctionNode(:*, copy_tree(step_node), 
                    SOFunctionNode(:/, SOVariable(:x), SOVariable(:z)))
        else_part = SOFunctionNode(:*, one_minus_step,
                    SOFunctionNode(:/, SOVariable(:y), SOVariable(:z)))
        tree = SOFunctionNode(:+, if_part, else_part)
        
        result = simplify_piecewise(tree)
        
        @test result isa PiecewiseResult
        @test contains(result.condition_string, "x") && contains(result.condition_string, "y")
        @test !isempty(result.if_string)
        @test !isempty(result.else_string)
        @test !isempty(result.if_latex)
        @test !isempty(result.else_latex)
        
        # Error when no step_func present
        simple_tree = SOFunctionNode(:+, SOVariable(:x), SOVariable(:y))
        @test_throws ErrorException simplify_piecewise(simple_tree)
    end
end
