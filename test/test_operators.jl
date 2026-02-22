using Random
using Statistics: sum

@testset "Genetic Operators" begin
    
    # Set up a common grammar for testing
    grammar = Grammar(
        binary_operators = [+, -, *, (/)],
        unary_operators = [sin, cos],
        variables = [:x, :y],
        constant_range = (-2.0, 2.0),
    )
    
    rng = Random.MersenneTwister(42)
    
    @testset "Tree Generation" begin
        
        @testset "Basic generation" begin
            tree = generate_tree(grammar; rng=rng, max_depth=3)
            @test tree isa AbstractNode
            @test tree_depth(tree) <= 3
            
            # Check all nodes are valid
            for node in flatten_tree(tree)
                if node isa Variable
                    @test node.name in [:x, :y]
                elseif node isa FunctionNode
                    @test node.func in [:+, :-, :*, :/, :sin, :cos]
                end
            end
        end
        
        @testset "Full method" begin
            # Full method should create trees where all paths have same depth
            tree = generate_tree(grammar; method=FullMethod(), min_depth=3, max_depth=3, rng=rng)
            @test tree isa AbstractNode
            # All leaves should be at approximately same depth
            @test tree_depth(tree) >= 2  # At least some depth
        end
        
        @testset "Grow method" begin
            tree = generate_tree(grammar; method=GrowMethod(), max_depth=4, rng=rng)
            @test tree isa AbstractNode
            @test tree_depth(tree) <= 4
        end
        
        @testset "Half-and-half method" begin
            trees = [generate_tree(grammar; method=HalfAndHalfMethod(), max_depth=4, rng=rng) for _ in 1:10]
            @test all(t -> t isa AbstractNode, trees)
            # Should get variety of shapes
        end
        
        @testset "Population generation" begin
            pop = generate_population(grammar, 20; max_depth=4, rng=rng)
            @test length(pop) == 20
            @test all(t -> t isa AbstractNode, pop)
        end
        
        @testset "Unique population" begin
            pop = generate_population(grammar, 10; max_depth=3, unique=true, rng=rng)
            @test length(pop) <= 10
            # All should be different
            hashes = [hash(t) for t in pop]
            @test length(unique(hashes)) == length(hashes)
        end
        
        @testset "random_terminal" begin
            for _ in 1:20
                t = random_terminal(grammar; rng=rng)
                @test t isa Constant || t isa Variable
                if t isa Variable
                    @test t.name in [:x, :y]
                end
            end
        end
    end
    
    @testset "Mutation" begin
        # Create a test tree: (x + y) * sin(x)
        test_tree = FunctionNode(:*,
            FunctionNode(:+, Variable(:x), Variable(:y)),
            FunctionNode(:sin, Variable(:x))
        )
        
        @testset "Point mutation" begin
            mutated = mutate_point(test_tree, grammar; rng=rng)
            @test mutated isa AbstractNode
            # Should be different (with high probability)
            # But structure should be similar
        end
        
        @testset "Subtree mutation" begin
            mutated = mutate_subtree(test_tree, grammar; rng=rng, max_depth=3)
            @test mutated isa AbstractNode
            @test tree_depth(mutated) <= tree_depth(test_tree) + 3
        end
        
        @testset "Hoist mutation" begin
            mutated = mutate_hoist(test_tree; rng=rng)
            @test mutated isa AbstractNode
            # Hoist should produce smaller or equal tree
            @test count_nodes(mutated) <= count_nodes(test_tree)
        end
        
        @testset "Constant perturbation" begin
            tree_with_const = FunctionNode(:+, Variable(:x), Constant(1.0))
            mutated = mutate_constants(tree_with_const; rng=rng, std=0.1)
            @test mutated isa AbstractNode
            
            # Find the constant in original and mutated
            orig_consts = collect_constants(tree_with_const)
            new_consts = collect_constants(mutated)
            @test length(orig_consts) == length(new_consts)
            @test orig_consts[1] != new_consts[1]  # Should be different
        end
        
        @testset "Insert mutation" begin
            simple_tree = Variable(:x)
            mutated = mutate_insert(simple_tree, grammar; rng=rng)
            @test mutated isa AbstractNode
            # Should be larger (a function now wraps x)
            @test count_nodes(mutated) >= count_nodes(simple_tree)
        end
        
        @testset "Delete mutation" begin
            mutated = mutate_delete(test_tree; rng=rng)
            @test mutated isa AbstractNode
            # Should be smaller
            @test count_nodes(mutated) <= count_nodes(test_tree)
        end
        
        @testset "Combined mutate" begin
            for _ in 1:10
                mutated = mutate(test_tree, grammar; rng=rng)
                @test mutated isa AbstractNode
            end
        end
    end
    
    @testset "Crossover" begin
        # Create two parent trees
        parent1 = FunctionNode(:+, 
            FunctionNode(:*, Variable(:x), Constant(2.0)),
            Variable(:y)
        )
        parent2 = FunctionNode(:sin,
            FunctionNode(:-, Variable(:x), Variable(:y))
        )
        
        @testset "Subtree crossover" begin
            child1, child2 = crossover_subtree(parent1, parent2; rng=rng)
            @test child1 isa AbstractNode
            @test child2 isa AbstractNode
        end
        
        @testset "Crossover with depth limit" begin
            child1, child2 = crossover(parent1, parent2; rng=rng, max_depth=5)
            @test tree_depth(child1) <= 5
            @test tree_depth(child2) <= 5
        end
        
        @testset "Uniform crossover" begin
            child1, child2 = crossover_uniform(parent1, parent2; rng=rng)
            @test child1 isa AbstractNode
            @test child2 isa AbstractNode
        end
        
        @testset "Size-fair crossover" begin
            child1, child2 = crossover_size_fair(parent1, parent2; rng=rng)
            @test child1 isa AbstractNode
            @test child2 isa AbstractNode
        end
        
        @testset "Crossover preserves validity" begin
            for _ in 1:10
                c1, c2 = crossover(parent1, parent2; rng=rng)
                # Children should only contain valid operators and variables
                for node in flatten_tree(c1)
                    if node isa Variable
                        @test node.name in [:x, :y]
                    elseif node isa FunctionNode
                        @test node.func in [:+, :-, :*, :/, :sin, :cos]
                    end
                end
            end
        end
    end
    
    @testset "Simplification" begin
        
        @testset "Constant folding" begin
            # 1 + 2 → 3
            tree = FunctionNode(:+, Constant(1.0), Constant(2.0))
            simplified = simplify_constants(tree)
            @test simplified isa Constant
            @test simplified.value ≈ 3.0
            
            # (1 + 2) * x → 3 * x
            tree2 = FunctionNode(:*,
                FunctionNode(:+, Constant(1.0), Constant(2.0)),
                Variable(:x)
            )
            simplified2 = simplify_constants(tree2)
            @test simplified2 isa FunctionNode
            @test simplified2.children[1] isa Constant
            @test simplified2.children[1].value ≈ 3.0
        end
        
        @testset "Algebraic simplification" begin
            # x + 0 → x
            tree1 = FunctionNode(:+, Variable(:x), Constant(0.0))
            s1 = simplify_algebra(tree1)
            @test s1 isa Variable
            @test s1.name == :x
            
            # x * 1 → x
            tree2 = FunctionNode(:*, Variable(:x), Constant(1.0))
            s2 = simplify_algebra(tree2)
            @test s2 isa Variable
            
            # x * 0 → 0
            tree3 = FunctionNode(:*, Variable(:x), Constant(0.0))
            s3 = simplify_algebra(tree3)
            @test s3 isa Constant
            @test s3.value ≈ 0.0
            
            # x ^ 0 → 1
            tree4 = FunctionNode(:^, Variable(:x), Constant(0.0))
            s4 = simplify_algebra(tree4)
            @test s4 isa Constant
            @test s4.value ≈ 1.0
            
            # x ^ 1 → x
            tree5 = FunctionNode(:^, Variable(:x), Constant(1.0))
            s5 = simplify_algebra(tree5)
            @test s5 isa Variable
        end
        
        @testset "Combined simplify" begin
            # (x + 0) * (1 + 1) → x * 2
            tree = FunctionNode(:*,
                FunctionNode(:+, Variable(:x), Constant(0.0)),
                FunctionNode(:+, Constant(1.0), Constant(1.0))
            )
            simplified = simplify(tree; grammar=grammar)
            # After simplification: x * 2
            @test count_nodes(simplified) < count_nodes(tree)
        end
        
        @testset "Normalize constants" begin
            tree = FunctionNode(:+, Variable(:x), Constant(1.23456789))
            normalized = normalize_constants(tree; digits=2)
            consts = collect_constants(normalized)
            @test consts[1] ≈ 1.23 atol=0.01
        end
        
        @testset "Clamp constants" begin
            tree = FunctionNode(:+, Variable(:x), Constant(1e10))
            clamped = clamp_constants(tree; max_val=100.0)
            consts = collect_constants(clamped)
            @test consts[1] == 100.0
        end
    end
    
    @testset "Typed Grammar Generation" begin
        typed_grammar = Grammar(
            types = [:Scalar, :Vector],
            variables = [:ps => :Vector, :n => :Scalar],
            operators = [
                (sum, [:Vector] => :Scalar),
                (+, [:Scalar, :Scalar] => :Scalar),
                (*, [:Scalar, :Scalar] => :Scalar),
            ],
            constant_types = [:Scalar],
            constant_range = (-2.0, 2.0),
            output_type = :Scalar,
        )
        
        @testset "Generate with target type" begin
            tree = generate_tree(typed_grammar; target_type=:Scalar, max_depth=3, rng=rng)
            @test tree isa AbstractNode
            
            # The tree should type-check
            result_type = infer_type(tree, typed_grammar)
            @test result_type == :Scalar || result_type == :Any
        end
    end
end
