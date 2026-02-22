@testset "Evaluation Engine" begin
    
    @testset "EvalContext" begin
        # Construction
        ctx1 = EvalContext(x=1.0, y=2.0)
        @test ctx1[:x] == 1.0
        @test ctx1[:y] == 2.0
        
        ctx2 = EvalContext(Dict(:a => 10, :b => 20))
        @test ctx2[:a] == 10
        
        ctx3 = EvalContext(:p => 0.5, :q => 0.7)
        @test ctx3[:p] == 0.5
        
        # Access
        @test haskey(ctx1, :x)
        @test !haskey(ctx1, :z)
        @test length(ctx1) == 2
        @test Set(keys(ctx1)) == Set([:x, :y])
        
        # Mutation
        ctx1[:z] = 3.0
        @test ctx1[:z] == 3.0
        
        # Merge
        ctx4 = merge(ctx1, EvalContext(w=4.0))
        @test ctx4[:x] == 1.0
        @test ctx4[:w] == 4.0
        
        ctx5 = merge(ctx1, w=5.0)
        @test ctx5[:w] == 5.0
    end
    
    @testset "Basic Evaluation" begin
        # Constant
        c = Constant(42.0)
        @test evaluate(c, EvalContext()) == 42.0
        
        # Variable
        v = Variable(:x)
        @test evaluate(v, x=5.0) == 5.0
        @test evaluate(v, EvalContext(x=3.0)) == 3.0
        
        # Simple addition
        tree = FunctionNode(:+, Variable(:x), Constant(1.0))
        @test evaluate(tree, x=5.0) == 6.0
        @test evaluate(tree, x=0.0) == 1.0
        @test evaluate(tree, x=-3.0) == -2.0
        
        # Nested expression: (x + y) * 2
        tree2 = FunctionNode(:*, 
            FunctionNode(:+, Variable(:x), Variable(:y)),
            Constant(2.0)
        )
        @test evaluate(tree2, x=3.0, y=4.0) == 14.0
        
        # Unary function: sin(x)
        tree3 = FunctionNode(:sin, Variable(:x))
        @test evaluate(tree3, x=0.0) ≈ 0.0
        @test evaluate(tree3, x=π/2) ≈ 1.0
    end
    
    @testset "Safe Operations" begin
        # Division by zero
        tree = FunctionNode(:/, Variable(:x), Variable(:y))
        @test isnan(evaluate(tree, x=1.0, y=0.0))
        
        # Log of negative
        tree2 = FunctionNode(:log, Variable(:x))
        @test isnan(evaluate(tree2, x=-1.0))
        @test evaluate(tree2, x=1.0) ≈ 0.0
        
        # Sqrt of negative
        tree3 = FunctionNode(:sqrt, Variable(:x))
        @test isnan(evaluate(tree3, x=-1.0))
        @test evaluate(tree3, x=4.0) ≈ 2.0
        
        # Power with negative base, fractional exponent
        tree4 = FunctionNode(:^, Variable(:x), Constant(0.5))
        @test isnan(evaluate(tree4, x=-4.0))
        @test evaluate(tree4, x=4.0) ≈ 2.0
    end
    
    @testset "Evaluation with Grammar" begin
        g = Grammar(
            binary_operators = [+, -, *, (/)],
            unary_operators = [sin, cos],
            variables = [:x, :y],
        )
        
        tree = FunctionNode(:+, 
            FunctionNode(:sin, Variable(:x)),
            Variable(:y)
        )
        
        ctx = EvalContext(x=0.0, y=1.0)
        @test evaluate(tree, g, ctx) ≈ 1.0  # sin(0) + 1
        
        ctx2 = EvalContext(x=π/2, y=0.0)
        @test evaluate(tree, g, ctx2) ≈ 1.0  # sin(π/2) + 0
    end
    
    @testset "Batch Evaluation" begin
        tree = FunctionNode(:+, Variable(:x), Variable(:y))
        
        # Matrix form
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        results = evaluate_batch(tree, data, [:x, :y])
        @test results == [3.0, 7.0, 11.0]
        
        # Vector of Dicts
        data2 = [
            Dict(:x => 1.0, :y => 2.0),
            Dict(:x => 10.0, :y => 20.0),
        ]
        results2 = evaluate_batch(tree, data2)
        @test results2 == [3.0, 30.0]
    end
    
    @testset "Safe Evaluate" begin
        # Normal case
        tree = FunctionNode(:+, Variable(:x), Constant(1.0))
        @test safe_evaluate(tree, EvalContext(x=5.0)) == 6.0
        
        # Error case (missing variable)
        @test isnan(safe_evaluate(tree, EvalContext(y=5.0)))
        
        # Division by zero
        tree2 = FunctionNode(:/, Constant(1.0), Variable(:x))
        @test isnan(safe_evaluate(tree2, EvalContext(x=0.0)))
    end
    
    @testset "is_valid_on" begin
        tree = FunctionNode(:sqrt, Variable(:x))
        
        @test is_valid_on(tree, EvalContext(x=4.0))
        @test !is_valid_on(tree, EvalContext(x=-4.0))
    end
    
    @testset "Complex Expressions" begin
        # Polynomial: x^2 + 2x + 1
        tree = FunctionNode(:+,
            FunctionNode(:+,
                FunctionNode(:^, Variable(:x), Constant(2.0)),
                FunctionNode(:*, Constant(2.0), Variable(:x))
            ),
            Constant(1.0)
        )
        
        @test evaluate(tree, x=0.0) ≈ 1.0
        @test evaluate(tree, x=1.0) ≈ 4.0   # 1 + 2 + 1
        @test evaluate(tree, x=2.0) ≈ 9.0   # 4 + 4 + 1
        @test evaluate(tree, x=-1.0) ≈ 0.0  # 1 - 2 + 1
        
        # Trigonometric: sin(x)^2 + cos(x)^2 = 1
        tree2 = FunctionNode(:+,
            FunctionNode(:^, FunctionNode(:sin, Variable(:x)), Constant(2.0)),
            FunctionNode(:^, FunctionNode(:cos, Variable(:x)), Constant(2.0))
        )
        
        for x in [0.0, 0.5, 1.0, π/4, π/2, π]
            @test evaluate(tree2, x=x) ≈ 1.0 atol=1e-10
        end
    end
    
    @testset "Compile Tree" begin
        tree = FunctionNode(:*,
            FunctionNode(:+, Variable(:x), Constant(1.0)),
            Variable(:y)
        )
        
        f = compile_tree(tree, [:x, :y])
        
        @test f(2.0, 3.0) == 9.0   # (2 + 1) * 3
        @test f(0.0, 5.0) == 5.0   # (0 + 1) * 5
        @test f(-1.0, 2.0) == 0.0  # (-1 + 1) * 2
        
        # Compiled should be faster for repeated evaluation
        # (not testing speed, just correctness)
        for _ in 1:100
            x, y = rand(), rand()
            @test f(x, y) ≈ evaluate(tree, x=x, y=y)
        end
    end
end
