@testset "Printing" begin
    
    # Build test trees
    x = Variable(:x)
    y = Variable(:y)
    c1 = Constant(1.0)
    c2 = Constant(2.5)
    
    @testset "node_to_string" begin
        # Constants
        @test node_to_string(Constant(1.0)) == "1"
        @test node_to_string(Constant(3.14159)) == "3.142"
        @test node_to_string(Constant(3.14159); digits=5) == "3.14159"
        
        # Variables
        @test node_to_string(x) == "x"
        @test node_to_string(Variable(:ps)) == "ps"
        
        # Simple infix
        add = FunctionNode(:+, x, c1)
        @test occursin("x", node_to_string(add))
        @test occursin("+", node_to_string(add))
        @test occursin("1", node_to_string(add))
        
        # Prefix functions
        sin_x = FunctionNode(:sin, x)
        @test node_to_string(sin_x) == "sin(x)"
        
        # Nested
        nested = FunctionNode(:sin, FunctionNode(:+, x, c1))
        str = node_to_string(nested)
        @test occursin("sin", str)
        @test occursin("+", str)
    end
    
    @testset "node_to_latex" begin
        # Constants
        @test node_to_latex(Constant(1.0)) == "1"
        @test node_to_latex(Constant(3.14)) == "3.14"
        
        # Variables
        @test node_to_latex(x) == "x"
        @test node_to_latex(Variable(:alpha)) == "\\alpha"
        @test node_to_latex(Variable(:beta)) == "\\beta"
        
        # Division becomes fraction
        div_tree = FunctionNode(:/, x, y)
        latex = node_to_latex(div_tree)
        @test occursin("\\frac", latex)
        @test occursin("x", latex)
        @test occursin("y", latex)
        
        # Power
        pow_tree = FunctionNode(:^, x, c2)
        latex = node_to_latex(pow_tree)
        @test occursin("^", latex)
        @test occursin("2.5", latex)
        
        # Sqrt
        sqrt_tree = FunctionNode(:sqrt, x)
        @test occursin("\\sqrt", node_to_latex(sqrt_tree))
        
        # Absolute value
        abs_tree = FunctionNode(:abs, x)
        latex = node_to_latex(abs_tree)
        @test occursin("\\left|", latex)
        @test occursin("\\right|", latex)
        
        # Multiplication uses cdot
        mul_tree = FunctionNode(:*, x, y)
        @test occursin("\\cdot", node_to_latex(mul_tree))
    end
    
    @testset "show methods" begin
        # Test that show methods don't error
        io = IOBuffer()
        
        show(io, Constant(1.0))
        @test occursin("Constant", String(take!(io)))
        
        show(io, Variable(:x))
        @test occursin("Variable", String(take!(io)))
        
        show(io, Variable(:x, :Scalar))
        str = String(take!(io))
        @test occursin("Variable", str)
        @test occursin("Scalar", str)
        
        tree = FunctionNode(:+, x, c1)
        show(io, tree)
        @test length(take!(io)) > 0
        
        # MIME text/plain
        show(io, MIME("text/plain"), tree)
        str = String(take!(io))
        @test occursin("Expression tree", str)
        @test occursin("nodes", str)
        @test occursin("depth", str)
    end
    
    @testset "print_tree" begin
        tree = FunctionNode(:+, x, FunctionNode(:*, y, c1))
        
        io = IOBuffer()
        print_tree(tree; io=io)
        output = String(take!(io))
        
        # Should have hierarchical structure
        @test occursin("+", output)
        @test occursin("x", output)
        @test occursin("*", output)
        @test occursin("y", output)
        @test occursin("1", output)
        
        # Children should be indented
        lines = split(output, "\n")
        @test length(lines) >= 5  # At least 5 nodes
    end
    
    @testset "tree_to_string_block" begin
        tree = FunctionNode(:sin, x)
        block = tree_to_string_block(tree)
        
        @test occursin("sin", block)
        @test occursin("x", block)
    end
end
