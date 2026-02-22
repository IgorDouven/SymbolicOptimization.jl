using Test
using SymbolicOptimization

@testset "API Module" begin

    @testset "SymbolicModel construction" begin
        m = SymbolicModel()
        @test m.status == :unsolved
        @test m.population_size == 200
        @test m.max_generations == 100
        @test isempty(m.variable_names)
        @test m.binary_ops == Any[+, -, *, /]
        @test isempty(m.objectives)

        m2 = SymbolicModel(seed=42, verbose=false)
        @test m2.random_seed == 42
        @test m2.verbose == false
    end

    @testset "Functional API - set_variables!" begin
        m = SymbolicModel()
        set_variables!(m, :x, :y, :z)
        @test m.variable_names == [:x, :y, :z]
    end

    @testset "Functional API - set_operators!" begin
        m = SymbolicModel()
        set_operators!(m, binary=[+, -], unary=[sin, cos])
        @test m.binary_ops == Any[+, -]
        @test m.unary_ops == Any[sin, cos]

        set_operators!(m, ternary=[safe_ifelse])
        @test m.ternary_ops == Any[safe_ifelse]
    end

    @testset "Functional API - set_constants!" begin
        m = SymbolicModel()
        set_constants!(m, -5.0, 5.0, probability=0.5)
        @test m.constant_range == (-5.0, 5.0)
        @test m.constant_prob == 0.5

        set_constants!(m, nothing)
        @test m.constant_range === nothing
        @test m.constant_prob == 0.0
    end

    @testset "Functional API - add_objective!" begin
        m = SymbolicModel()
        add_objective!(m, :Min, :mse)
        add_objective!(m, :Min, :complexity)
        @test length(m.objectives) == 2
        @test m.objectives[1] == :mse
        @test m.objectives[2] == :complexity

        # Custom objective
        add_objective!(m, :Max, :custom_obj, (tree, env) -> 1.0)
        @test length(m.objectives) == 3
        name, sense, func = m.objectives[3]
        @test name == :custom_obj
        @test sense == :Max

        # Bad built-in should error
        @test_throws ErrorException add_objective!(m, :Min, :unknown_builtin)

        # Bad sense should error
        @test_throws AssertionError add_objective!(m, :Minimize, :mse)
    end

    @testset "Functional API - add_constraint!" begin
        m = SymbolicModel()
        add_constraint!(m, :neutrality, (tree, eval_fn) -> (true, 0.0),
                        description="Test constraint")
        @test length(m.constraints) == 1
        @test m.constraints[1].name == :neutrality
        @test m.constraints[1].description == "Test constraint"

        set_constraint_mode!(m, :hard, penalty_weight=2.0)
        @test m.constraint_mode == :hard
        @test m.constraint_penalty == 2.0
    end

    @testset "Functional API - set_data! and set_config!" begin
        m = SymbolicModel()
        X = rand(10, 2)
        y = rand(10)
        set_data!(m, X=X, y=y, extra=42)
        @test m.environment[:X] === X
        @test m.environment[:y] === y
        @test m.environment[:extra] == 42

        set_config!(m, population=500, generations=200, max_depth=10)
        @test m.population_size == 500
        @test m.max_generations == 200
        @test m.max_depth == 10
    end

    @testset "expr_to_tree" begin
        vars = Set([:x, :y, :pH, :pE, :pH_E, :pE_H])

        # Simple variable
        tree = expr_to_tree(:x, vars)
        @test tree isa Variable
        @test tree.name == :x

        # Numeric constant
        tree = expr_to_tree(3.14, vars)
        @test tree isa Constant
        @test tree.value ≈ 3.14

        # Integer constant
        tree = expr_to_tree(2, vars)
        @test tree isa Constant
        @test tree.value ≈ 2.0

        # Binary operation: x + y
        tree = expr_to_tree(:(x + y), vars)
        @test tree isa FunctionNode
        @test tree.func == :+
        @test length(tree.children) == 2
        @test tree.children[1] isa Variable
        @test tree.children[1].name == :x

        # Nested: pH_E - pE
        tree = expr_to_tree(:(pH_E - pE), vars)
        @test tree isa FunctionNode
        @test tree.func == :-
        @test tree.children[1].name == :pH_E
        @test tree.children[2].name == :pE

        # Unary function: log(x)
        tree = expr_to_tree(:(log(x)), vars)
        @test tree isa FunctionNode
        @test tree.func == :log
        @test length(tree.children) == 1
        @test tree.children[1].name == :x

        # Compound: log(pE_H / pH_E)
        tree = expr_to_tree(:(log(pE_H / pH_E)), vars)
        @test tree isa FunctionNode
        @test tree.func == :log
        div_node = tree.children[1]
        @test div_node isa FunctionNode
        @test div_node.func == :/
        @test div_node.children[1].name == :pE_H
        @test div_node.children[2].name == :pH_E

        # With constant: (pH_E - pE) / (1.0 - pE)
        tree = expr_to_tree(:((pH_E - pE) / (1.0 - pE)), vars)
        @test tree isa FunctionNode
        @test tree.func == :/
        num = tree.children[1]
        @test num.func == :-
        den = tree.children[2]
        @test den.func == :-
        @test den.children[1] isa Constant
        @test den.children[1].value ≈ 1.0
    end

    @testset "add_seed!" begin
        m = SymbolicModel()
        set_variables!(m, :x, :y)

        tree = FunctionNode(:+, [Variable(:x), Variable(:y)])
        add_seed!(m, tree)
        @test length(m.seed_trees) == 1
        @test m.seed_trees[1] isa FunctionNode
    end

    @testset "Macro API - @variables" begin
        m = SymbolicModel()
        @variables(m, a, b, c)
        @test m.variable_names == [:a, :b, :c]
    end

    @testset "Macro API - @seed" begin
        m = SymbolicModel()
        @variables(m, pH, pE, pH_E, pE_H)

        @seed(m, pH_E - pH)
        @test length(m.seed_trees) == 1
        tree = m.seed_trees[1]
        @test tree isa FunctionNode
        @test tree.func == :-
        @test tree.children[1].name == :pH_E
        @test tree.children[2].name == :pH

        @seed(m, log(pE_H / pH_E))
        @test length(m.seed_trees) == 2
        tree2 = m.seed_trees[2]
        @test tree2.func == :log
        @test tree2.children[1].func == :/
    end

    @testset "End-to-end: simple regression" begin
        # y = x² (simple case)
        X = reshape(range(-2, 2, length=20), :, 1)
        y = vec(X.^2)

        m = SymbolicModel()
        @variables(m, x)
        @operators(m, binary=[+, -, *, /])
        @constants(m, -2.0..2.0)
        @objective(m, Min, :mse)
        @objective(m, Min, :complexity)
        @data(m, X=X, y=y)
        @config(m, population=100, generations=30, seed=42, verbose=false)

        optimize!(m)

        @test m.status == :optimal
        @test m.result !== nothing

        b = best(m)
        @test b.expression isa String
        @test length(b.objectives) == 2
        @test b.objectives[1] >= 0  # MSE non-negative
        @test b.objectives[2] >= 1  # At least 1 node

        front = pareto_front(m)
        @test length(front) >= 1
        @test all(sol -> sol.expression isa String, front)

        # Predict
        X_test = reshape([0.0, 1.0, -1.0], :, 1)
        preds = predict(m, X_test)
        @test length(preds) == 3

        # Other accessors
        @test objective_value(m) >= 0
        @test expression_string(m) isa String
        @test expression_latex(m) isa String
        @test history(m) isa Vector
        @test raw_result(m) isa NSGAIIResult
    end

    @testset "End-to-end: custom objective" begin
        X = reshape(range(-2, 2, length=20), :, 1)
        y = vec(X.^2)

        m = SymbolicModel()
        @variables(m, x)
        @operators(m, binary=[+, -, *, /])

        # Custom MAE objective
        @objective(m, Min, :custom_mae, (tree, env) -> begin
            X = env[:X]
            y = env[:y]
            var_names = env[:var_names]
            preds = evaluate_batch(tree, X, var_names)
            valid = .!isnan.(preds)
            sum(valid) == 0 && return Inf
            return sum(abs.(preds[valid] .- y[valid])) / sum(valid)
        end)
        @objective(m, Min, :complexity)

        @data(m, X=X, y=y)
        @config(m, population=50, generations=10, seed=42, verbose=false)

        optimize!(m)
        @test m.status == :optimal
    end

    @testset "Error handling" begin
        m = SymbolicModel()

        # Cannot get results before solving
        @test_throws ErrorException best(m)
        @test_throws ErrorException pareto_front(m)
        @test_throws ErrorException predict(m, rand(5, 1))

        # No objectives
        @variables(m, x)
        @data(m, X=rand(10, 1), y=rand(10))
        @test_throws ErrorException optimize!(m)

        # No variables and no X data
        m2 = SymbolicModel()
        add_objective!(m2, :Min, :mse)
        @test_throws ErrorException optimize!(m2)
    end

    @testset "Auto-infer variables from X data" begin
        m = SymbolicModel()
        add_objective!(m, :Min, :mse)
        add_objective!(m, :Min, :complexity)
        set_data!(m, X=rand(10, 3), y=rand(10))
        set_config!(m, population=20, generations=2, verbose=false)
        optimize!(m)
        @test m.variable_names == [:x1, :x2, :x3]
    end

    @testset "Display" begin
        m = SymbolicModel()
        @variables(m, x, y)
        @objective(m, Min, :mse)
        @objective(m, Min, :complexity)

        str = sprint(show, m)
        @test contains(str, "SymbolicModel")
        @test contains(str, "2 vars")

        detailed = sprint(show, MIME"text/plain"(), m)
        @test contains(detailed, "Variables")
        @test contains(detailed, "Objectives")
    end

end
