using Random

@testset "NSGA-II Optimization" begin
    
    rng = Random.MersenneTwister(42)
    
    @testset "Individual" begin
        tree = FunctionNode(:+, Variable(:x), Constant(1.0))
        
        ind = Individual(tree)
        @test ind.tree isa AbstractNode
        @test isempty(ind.objectives)
        @test ind.rank == 0
        @test ind.crowding_distance == 0.0
        
        ind2 = Individual(tree, [1.0, 2.0])
        @test ind2.objectives == [1.0, 2.0]
        
        # Copy
        ind_copy = copy(ind2)
        @test ind_copy.objectives == ind2.objectives
        @test ind_copy.tree !== ind2.tree  # Deep copy
    end
    
    @testset "Objective Functions" begin
        # Built-in objectives
        mse_obj = mse_objective()
        @test mse_obj.minimize == true
        @test mse_obj.name == :mse
        
        complexity_obj = complexity_objective()
        @test complexity_obj.minimize == true
        
        # Custom objective
        custom_obj = custom_objective(:test, (tree, data) -> 1.0; minimize=false)
        @test custom_obj.minimize == false
        @test custom_obj.name == :test
    end
    
    @testset "Pareto Dominance" begin
        ind1 = Individual(Variable(:x), [1.0, 2.0])
        ind2 = Individual(Variable(:x), [2.0, 3.0])
        ind3 = Individual(Variable(:x), [1.0, 3.0])
        ind4 = Individual(Variable(:x), [2.0, 2.0])
        
        minimize = [true, true]
        
        # ind1 dominates ind2 (better in both)
        @test dominates(ind1, ind2, minimize)
        @test !dominates(ind2, ind1, minimize)
        
        # ind1 dominates ind3 (same in first, better in second)
        @test dominates(ind1, ind3, minimize)
        
        # ind1 does NOT dominate ind4 (same in second, worse in... wait, ind1 is 1,2 and ind4 is 2,2)
        # ind1 is better in first objective (1 < 2), same in second (2 = 2)
        @test dominates(ind1, ind4, minimize)
        
        # Neither dominates the other when trade-off
        ind5 = Individual(Variable(:x), [0.5, 3.5])  # Better in first, worse in second
        @test !dominates(ind1, ind5, minimize)
        @test !dominates(ind5, ind1, minimize)
    end
    
    @testset "Non-dominated Sorting" begin
        # Create population with clear Pareto structure
        pop = [
            Individual(Variable(:x), [1.0, 4.0]),  # Front 1
            Individual(Variable(:x), [2.0, 2.0]),  # Front 1
            Individual(Variable(:x), [4.0, 1.0]),  # Front 1
            Individual(Variable(:x), [3.0, 3.0]),  # Front 2 (dominated by [2,2])
            Individual(Variable(:x), [5.0, 5.0]),  # Front 3
        ]
        
        objectives = [mse_objective(), complexity_objective()]
        fronts = nondominated_sort!(pop, objectives)
        
        @test length(fronts) >= 2
        @test pop[1].rank == 1
        @test pop[2].rank == 1
        @test pop[3].rank == 1
        @test pop[4].rank == 2
        @test pop[5].rank >= 2
    end
    
    @testset "Crowding Distance" begin
        pop = [
            Individual(Variable(:x), [1.0, 4.0]),
            Individual(Variable(:x), [2.0, 2.0]),
            Individual(Variable(:x), [4.0, 1.0]),
        ]
        
        # Assign rank 1 to all
        for ind in pop
            ind.rank = 1
        end
        
        front_indices = [1, 2, 3]
        compute_crowding_distance!(pop, front_indices)
        
        # Boundary points should have infinite distance
        @test isinf(pop[1].crowding_distance) || isinf(pop[3].crowding_distance)
        
        # Middle point should have finite distance
        @test isfinite(pop[2].crowding_distance)
    end
    
    @testset "Tournament Selection" begin
        pop = [
            Individual(Variable(:x), [1.0, 1.0]),
            Individual(Variable(:x), [2.0, 2.0]),
            Individual(Variable(:x), [3.0, 3.0]),
        ]
        
        objectives = [mse_objective(), complexity_objective()]
        nondominated_sort!(pop, objectives)
        
        # The best (rank 1) should be selected more often
        selections = [tournament_select(pop, 2; rng=rng) for _ in 1:20]
        
        # At least some selections should be the best individual
        @test any(s -> s.objectives == [1.0, 1.0], selections)
    end
    
    @testset "NSGAIIConfig" begin
        config = NSGAIIConfig()
        @test config.population_size == 100
        @test config.max_generations == 50
        
        config2 = NSGAIIConfig(population_size=50, max_generations=10)
        @test config2.population_size == 50
        @test config2.max_generations == 10
    end
    
    @testset "Basic Optimization" begin
        grammar = Grammar(
            binary_operators = [+, -, *],
            unary_operators = [sin],
            variables = [:x],
            constant_range = (-2.0, 2.0),
        )
        
        # Simple quadratic target: x^2
        x_data = collect(-2.0:0.5:2.0)
        y_data = x_data.^2
        
        data = Dict{Symbol,Any}(
            :X => reshape(x_data, :, 1),
            :y => y_data,
            :var_names => [:x],
        )
        
        objectives = [mse_objective(), complexity_objective()]
        
        config = NSGAIIConfig(
            population_size = 20,
            max_generations = 5,
            verbose = false,
        )
        
        result = optimize(grammar, objectives, data; config=config, rng=rng)
        
        @test result isa NSGAIIResult
        @test result.generations == 5
        @test !isempty(result.pareto_front)
        @test !isempty(result.population)
        @test length(result.best_per_objective) == 2
    end
    
    @testset "Curve Fitting Convenience" begin
        x = collect(-2.0:0.5:2.0)
        y = x.^2 .+ x  # Simple quadratic
        
        config = NSGAIIConfig(
            population_size = 15,
            max_generations = 3,
            verbose = false,
        )
        
        result = curve_fitting(x, y; config=config, rng=rng)
        
        @test result isa NSGAIIResult
        @test !isempty(result.pareto_front)
        
        # Get best
        best = get_best(result, 1)
        @test best isa Individual
        @test !isempty(best.objectives)
        
        # Test alias still works
        result2 = symbolic_regression(x, y; config=config, rng=rng)
        @test result2 isa NSGAIIResult
    end
    
    @testset "Custom Objectives" begin
        grammar = Grammar(
            binary_operators = [+, *],
            variables = [:x],
        )
        
        # Custom objective: penalize use of multiplication
        function count_mults(tree::AbstractNode, data::Dict)
            count = 0
            for node in flatten_tree(tree)
                if node isa FunctionNode && node.func == :*
                    count += 1
                end
            end
            return Float64(count)
        end
        
        mult_penalty = custom_objective(:mult_count, count_mults; minimize=true)
        
        x_data = collect(-1.0:0.5:1.0)
        data = Dict{Symbol,Any}(
            :X => reshape(x_data, :, 1),
            :y => x_data .+ 1.0,
            :var_names => [:x],
        )
        
        objectives = [mse_objective(), mult_penalty]
        
        config = NSGAIIConfig(
            population_size = 10,
            max_generations = 2,
            verbose = false,
        )
        
        result = optimize(grammar, objectives, data; config=config, rng=rng)
        @test result isa NSGAIIResult
    end
    
    @testset "Environmental Selection" begin
        # Test that environmental selection keeps the right individuals
        pop = [
            Individual(Variable(:x), [1.0, 1.0]),  # Best
            Individual(Variable(:x), [2.0, 2.0]),
            Individual(Variable(:x), [3.0, 3.0]),
            Individual(Variable(:x), [4.0, 4.0]),
            Individual(Variable(:x), [5.0, 5.0]),  # Worst
        ]
        
        objectives = [mse_objective(), complexity_objective()]
        
        selected = environmental_select!(pop, 3, objectives)
        
        @test length(selected) == 3
        # Best individual should be selected
        @test any(s -> s.objectives == [1.0, 1.0], selected)
    end
end
