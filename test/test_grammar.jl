using Statistics: mean, sum

@testset "Grammar System" begin
    
    @testset "Simple Grammar Construction" begin
        # Default grammar
        g = Grammar()
        @test !is_typed(g)
        @test num_operators(g) == 4  # +, -, *, /
        @test num_variables(g) == 1  # x
        
        # Custom simple grammar
        g2 = Grammar(
            binary_operators = [+, -, *],
            unary_operators = [sin, cos, exp],
            variables = [:x, :y, :z],
            constant_range = (-1.0, 1.0),
        )
        
        @test !is_typed(g2)
        @test num_operators(g2) == 6  # 3 binary + 3 unary
        @test num_variables(g2) == 3
        @test length(unary_operators(g2)) == 3
        @test length(binary_operators(g2)) == 3
        
        # Check operator names
        @test has_operator(g2, :+)
        @test has_operator(g2, :sin)
        @test !has_operator(g2, :tan)
    end
    
    @testset "Typed Grammar Construction" begin
        g = Grammar(
            types = [:Scalar, :Vector],
            variables = [:ps => :Vector, :n => :Scalar],
            operators = [
                (mean, [:Vector] => :Scalar),
                (sum, [:Vector] => :Scalar),
                (+, [:Scalar, :Scalar] => :Scalar),
                (*, [:Scalar, :Scalar] => :Scalar),
            ],
            constant_types = [:Scalar],
            output_type = :Scalar,
        )
        
        @test is_typed(g)
        @test has_type(g, :Scalar)
        @test has_type(g, :Vector)
        @test !has_type(g, :Matrix)
        
        @test num_operators(g) == 4
        @test num_variables(g) == 2
        
        # Check operators by type
        scalar_producers = operators_producing(g, :Scalar)
        @test length(scalar_producers) == 4  # mean, sum, +, *
        
        # Check variables by type
        vec_vars = variables_of_type(g, :Vector)
        @test length(vec_vars) == 1
        @test vec_vars[1].name == :ps
        
        scalar_vars = variables_of_type(g, :Scalar)
        @test length(scalar_vars) == 1
        @test scalar_vars[1].name == :n
    end
    
    @testset "Grammar Accessors" begin
        g = Grammar(
            binary_operators = [+, -, *, /],
            unary_operators = [sin, cos],
            variables = [:x, :y],
        )
        
        # By arity
        @test length(unary_operators(g)) == 2
        @test length(binary_operators(g)) == 4
        @test length(operators_by_arity(g, 3)) == 0
        
        # By name
        plus_ops = operators_by_name(g, :+)
        @test length(plus_ops) == 1
        @test plus_ops[1].arity == 2
        
        sin_ops = operators_by_name(g, :sin)
        @test length(sin_ops) == 1
        @test sin_ops[1].arity == 1
    end
    
    @testset "Constant Sampling" begin
        g = Grammar(
            variables = [:x],
            constant_range = (0.0, 10.0),
        )
        
        # Sample many constants and check range
        for _ in 1:100
            c = sample_constant(g)
            @test 0.0 <= c <= 10.0
        end
    end
    
    @testset "Complexity" begin
        g = Grammar(
            binary_operators = [+, (^)],
            unary_operators = [sin],
            variables = [:x],
            complexity_of_operators = Dict((^) => 3.0, sin => 2.0),
            complexity_of_variables = 0.5,
            complexity_of_constants = 1.5,
        )
        
        # Check operator complexity
        plus_op = operators_by_name(g, :+)[1]
        pow_op = operators_by_name(g, :^)[1]
        sin_op = operators_by_name(g, :sin)[1]
        
        @test plus_op.complexity == 1.0  # default
        @test pow_op.complexity == 3.0
        @test sin_op.complexity == 2.0
        
        # Check variable complexity
        @test g.variables[1].complexity == 0.5
        
        # Check constant complexity
        @test g.constants[1].complexity == 1.5
    end
    
    @testset "Grammar Display" begin
        g = Grammar(
            binary_operators = [+, *],
            unary_operators = [sin],
            variables = [:x, :y],
        )
        
        # Just check that show methods don't error
        io = IOBuffer()
        show(io, g)
        @test length(take!(io)) > 0
        
        show(io, MIME("text/plain"), g)
        output = String(take!(io))
        @test occursin("Grammar", output)
        @test occursin("Operators", output)
        @test occursin("Variables", output)
    end
end

@testset "Safe Operations" begin
    using SymbolicOptimization: safe_div, safe_pow, safe_log, safe_exp, safe_sqrt
    using SymbolicOptimization: safe_mean, safe_sum, safe_std, safe_var
    using SymbolicOptimization: sigmoid, clamp01
    
    @testset "safe_div" begin
        @test safe_div(6.0, 2.0) == 3.0
        @test isnan(safe_div(1.0, 0.0))
        @test isnan(safe_div(1.0, 1e-15))
    end
    
    @testset "safe_pow" begin
        @test safe_pow(2.0, 3.0) == 8.0
        @test isnan(safe_pow(-2.0, 0.5))  # sqrt of negative
        @test safe_pow(-2.0, 2.0) == 4.0  # integer exponent OK
        @test isfinite(safe_pow(2.0, 100.0))  # clamped exponent
    end
    
    @testset "safe_log" begin
        @test safe_log(exp(1.0)) ≈ 1.0
        @test isnan(safe_log(0.0))
        @test isnan(safe_log(-1.0))
    end
    
    @testset "safe_exp" begin
        @test safe_exp(0.0) == 1.0
        @test isfinite(safe_exp(1000.0))  # clamped
        @test isfinite(safe_exp(-1000.0))  # clamped
    end
    
    @testset "safe_sqrt" begin
        @test safe_sqrt(4.0) == 2.0
        @test isnan(safe_sqrt(-1.0))
    end
    
    @testset "reductions" begin
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        @test safe_mean(v) == 3.0
        @test safe_sum(v) == 15.0
        @test safe_std(v) > 0
        @test safe_var(v) > 0
        
        # Empty vector
        @test isnan(safe_mean(Float64[]))
        @test safe_sum(Float64[]) == 0.0
    end
    
    @testset "special functions" begin
        @test sigmoid(0.0) == 0.5
        @test 0.0 < sigmoid(-10.0) < 0.5
        @test 0.5 < sigmoid(10.0) < 1.0
        
        @test clamp01(-0.5) == 0.0
        @test clamp01(0.5) == 0.5
        @test clamp01(1.5) == 1.0
    end
end

@testset "Grammar Validation" begin
    @testset "Valid simple grammar" begin
        g = Grammar(
            binary_operators = [+, *],
            unary_operators = [sin],
            variables = [:x],
        )
        
        result = validate_grammar(g)
        @test result.valid
        @test isempty(result.errors)
    end
    
    @testset "Valid typed grammar" begin
        g = Grammar(
            types = [:Scalar, :Vector],
            variables = [:ps => :Vector, :n => :Scalar],
            operators = [
                (mean, [:Vector] => :Scalar),
                (+, [:Scalar, :Scalar] => :Scalar),
            ],
            constant_types = [:Scalar],
            output_type = :Scalar,
        )
        
        result = validate_grammar(g)
        @test result.valid
    end
    
    @testset "Grammar with no terminals" begin
        # This should error during construction or validation
        g = Grammar(
            binary_operators = [+],
            variables = Symbol[],
            constant_range = nothing,
        )
        
        result = validate_grammar(g)
        @test !result.valid
        @test any(e -> occursin("terminal", lowercase(e)), result.errors)
    end
    
    @testset "Tree validity check" begin
        g = Grammar(
            binary_operators = [+, *],
            unary_operators = [sin],
            variables = [:x, :y],
        )
        
        # Valid tree
        tree = FunctionNode(:+, Variable(:x), Constant(1.0))
        valid, msg = check_tree_validity(tree, g)
        @test valid
        
        # Invalid operator
        tree2 = FunctionNode(:tan, Variable(:x))
        valid2, msg2 = check_tree_validity(tree2, g)
        @test !valid2
        @test occursin("tan", msg2)
        
        # Invalid variable
        tree3 = FunctionNode(:+, Variable(:z), Constant(1.0))
        valid3, msg3 = check_tree_validity(tree3, g)
        @test !valid3
        @test occursin("z", msg3)
    end
end
