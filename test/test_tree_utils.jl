@testset "Tree Utilities" begin
    
    # Build some test trees
    x = Variable(:x)
    y = Variable(:y)
    c1 = Constant(1.0)
    c2 = Constant(2.0)
    
    # Simple: (x + 1)
    simple = FunctionNode(:+, x, c1)
    
    # Medium: ((x + 1) * y)
    medium = FunctionNode(:*, simple, y)
    
    # Complex: sin((x + 1) * y) + 2
    complex = FunctionNode(:+, FunctionNode(:sin, medium), c2)
    
    @testset "copy_tree" begin
        copied = copy_tree(complex)
        
        # Should be equal
        @test copied == complex
        
        # But not the same object
        @test copied !== complex
        
        # Deep copy - modifying original shouldn't affect copy
        # (Can't really test this without mutation, but we verify structure)
        @test count_nodes(copied) == count_nodes(complex)
        
        # Copy preserves types
        typed_var = Variable(:ps, :Vector)
        typed_copy = copy_tree(typed_var)
        @test vartype(typed_copy) == :Vector
    end
    
    @testset "count_nodes" begin
        @test count_nodes(x) == 1
        @test count_nodes(c1) == 1
        @test count_nodes(simple) == 3  # +, x, 1
        @test count_nodes(medium) == 5  # *, +, x, 1, y
        @test count_nodes(complex) == 8  # +, sin, *, +, x, 1, y, 2
    end
    
    @testset "tree_depth" begin
        @test tree_depth(x) == 1
        @test tree_depth(c1) == 1
        @test tree_depth(simple) == 2
        @test tree_depth(medium) == 3
        @test tree_depth(complex) == 5
    end
    
    @testset "flatten_tree" begin
        nodes = flatten_tree(simple)
        @test length(nodes) == 3
        @test nodes[1] == simple  # Root first
        @test nodes[2] == x
        @test nodes[3] == c1
        
        # Order is pre-order (root, then children left-to-right recursively)
        nodes_complex = flatten_tree(complex)
        @test length(nodes_complex) == 8
        @test nodes_complex[1].func == :+  # outer +
        @test nodes_complex[2].func == :sin
    end
    
    @testset "collect_variables" begin
        @test collect_variables(c1) == Set{Symbol}()
        @test collect_variables(x) == Set([:x])
        @test collect_variables(simple) == Set([:x])
        @test collect_variables(medium) == Set([:x, :y])
        @test collect_variables(complex) == Set([:x, :y])
        
        # Variable used multiple times
        dup = FunctionNode(:+, x, x)
        @test collect_variables(dup) == Set([:x])
    end
    
    @testset "collect_constants" begin
        @test collect_constants(x) == Float64[]
        @test collect_constants(c1) == [1.0]
        @test collect_constants(simple) == [1.0]
        @test collect_constants(medium) == [1.0]
        @test collect_constants(complex) == [1.0, 2.0]
        
        # Multiple of same constant
        dup_const = FunctionNode(:+, c1, c1)
        @test collect_constants(dup_const) == [1.0, 1.0]
    end
    
    @testset "collect_functions" begin
        @test collect_functions(x) == Symbol[]
        @test collect_functions(simple) == [:+]
        @test collect_functions(medium) == [:*, :+]
        @test collect_functions(complex) == [:+, :sin, :*, :+]
    end
    
    @testset "replace_subtree" begin
        # Replace variable with constant
        new_tree = replace_subtree(simple, x, c2)
        @test new_tree.func == :+
        @test new_tree.children[1] == c2
        @test new_tree.children[2] == c1
        
        # Replace subtree
        new_complex = replace_subtree(complex, medium, Variable(:z))
        @test count_nodes(new_complex) == 4  # +, sin, z, 2
        
        # Original unchanged
        @test count_nodes(complex) == 8
        
        # Replace root
        new_root = replace_subtree(simple, simple, x)
        @test new_root == x
    end
    
    @testset "map_tree" begin
        # Double all constants
        doubled = map_tree(simple) do node
            if node isa Constant
                Constant(node.value * 2)
            else
                node
            end
        end
        
        @test doubled.children[2].value == 2.0  # 1.0 * 2
        
        # Rename variables
        renamed = map_tree(medium) do node
            if node isa Variable && node.name == :x
                Variable(:z)
            else
                node
            end
        end
        
        @test :z in collect_variables(renamed)
        @test :x ∉ collect_variables(renamed)
    end
    
    @testset "get_subtree_at_index" begin
        # simple = (+ x 1)
        @test get_subtree_at_index(simple, 1) == simple
        @test get_subtree_at_index(simple, 2) == x
        @test get_subtree_at_index(simple, 3) == c1
        
        # Out of bounds
        @test_throws BoundsError get_subtree_at_index(simple, 0)
        @test_throws BoundsError get_subtree_at_index(simple, 4)
    end
    
    @testset "indexed_nodes" begin
        pairs = indexed_nodes(simple)
        @test length(pairs) == 3
        @test pairs[1] == (1, simple)
        @test pairs[2] == (2, x)
        @test pairs[3] == (3, c1)
    end
    
    @testset "random_subtree" begin
        using Random
        Random.seed!(42)
        
        # Should return some node from the tree
        for _ in 1:10
            node = random_subtree(complex)
            @test node in flatten_tree(complex)
        end
    end
    
    @testset "terminals and nonterminals" begin
        terms = terminals(complex)
        nonterms = nonterminals(complex)
        
        @test all(isterminal, terms)
        @test all(isfunction, nonterms)
        @test length(terms) + length(nonterms) == count_nodes(complex)
    end
    
    @testset "tree_size_stats" begin
        stats = tree_size_stats(complex)
        
        @test stats.nodes == 8
        @test stats.depth == 5
        @test stats.constants == 2
        @test stats.variables == 2
        @test stats.functions == 4
        @test stats.terminals == 4
        @test stats.unique_vars == 2
        @test stats.unique_funcs == 3  # +, sin, *
    end
end
