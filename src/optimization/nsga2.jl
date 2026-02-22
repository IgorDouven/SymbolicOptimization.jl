# ═══════════════════════════════════════════════════════════════════════════════
# NSGA-II Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

using Random

# ───────────────────────────────────────────────────────────────────────────────
# Main Optimization Function
# ───────────────────────────────────────────────────────────────────────────────

"""
    optimize(grammar::Grammar, objectives::Vector{ObjectiveFunction}, data::Dict;
             config::NSGAIIConfig=NSGAIIConfig(), rng=Random.GLOBAL_RNG) -> NSGAIIResult

Run NSGA-II multi-objective optimization to find expression trees.

# Arguments
- `grammar`: The grammar defining valid expressions
- `objectives`: Vector of objective functions to optimize
- `data`: Dictionary containing training data and metadata

# Keyword Arguments
- `config`: NSGA-II configuration
- `rng`: Random number generator
- `initial_population`: Optional initial population

# Returns
An `NSGAIIResult` containing the Pareto front and optimization history.

# Example

```julia
grammar = Grammar(
    binary_operators = [+, -, *, /],
    unary_operators = [sin, cos],
    variables = [:x],
)

objectives = [mse_objective(), complexity_objective()]

data = Dict(
    :X => reshape(collect(-2.0:0.1:2.0), :, 1),
    :y => [x^2 + x for x in -2.0:0.1:2.0],
    :var_names => [:x],
)

result = optimize(grammar, objectives, data; 
    config=NSGAIIConfig(population_size=50, max_generations=20))

# Get the Pareto front
front = get_pareto_front(result)

# Get best for first objective (MSE)
best = get_best(result, 1)
```
"""
function optimize(grammar::Grammar, objectives::Vector{ObjectiveFunction}, data::Dict;
                  config::NSGAIIConfig = NSGAIIConfig(),
                  rng::AbstractRNG = Random.GLOBAL_RNG,
                  initial_population::Union{Nothing, Vector{Individual}} = nothing)
    
    # Add grammar to data for evaluation
    data_with_grammar = merge(data, Dict(:grammar => grammar))
    
    # Initialize population
    if initial_population !== nothing
        # Filter seeds by max_nodes
        if config.max_nodes > 0
            population = filter(ind -> count_nodes(ind.tree) <= config.max_nodes, initial_population)
        else
            population = initial_population
        end
        if length(population) < config.population_size
            # Generate additional individuals
            additional = _create_initial_population(grammar, config.population_size - length(population), config, rng)
            append!(population, additional)
        end
    else
        population = _create_initial_population(grammar, config.population_size, config, rng)
    end
    
    # Evaluate initial population
    _evaluate_population!(population, objectives, data_with_grammar)
    
    # Initial sorting and crowding
    fronts = nondominated_sort!(population, objectives)
    compute_all_crowding_distances!(population, fronts)
    
    # History tracking
    history = Dict{Symbol,Any}[]
    
    # Track best for early stopping
    best_objectives = [obj.minimize ? Inf : -Inf for obj in objectives]
    generations_without_improvement = 0
    
    # Main evolution loop
    for gen in 1:config.max_generations
        # Create offspring
        offspring = _create_offspring(population, grammar, config, rng)
        
        # Evaluate offspring
        _evaluate_population!(offspring, objectives, data_with_grammar)
        
        # Combine parent and offspring populations
        combined = vcat(population, offspring)
        
        # Environmental selection
        population = environmental_select!(combined, config.population_size, objectives)
        
        # Re-sort and compute crowding for selected population
        fronts = nondominated_sort!(population, objectives)
        compute_all_crowding_distances!(population, fronts)
        
        # Record history
        gen_stats = _compute_generation_stats(population, objectives, gen)
        push!(history, gen_stats)
        
        # Check for improvement (for early stopping)
        improved = false
        for (i, obj) in enumerate(objectives)
            current_best = obj.minimize ? minimum(ind.objectives[i] for ind in population) : 
                                         maximum(ind.objectives[i] for ind in population)
            if obj.minimize ? (current_best < best_objectives[i]) : (current_best > best_objectives[i])
                best_objectives[i] = current_best
                improved = true
            end
        end
        
        if improved
            generations_without_improvement = 0
        else
            generations_without_improvement += 1
        end
        
        # Verbose output
        if config.verbose && (gen == 1 || gen % 5 == 0 || gen == config.max_generations)
            _print_generation_stats(gen_stats, objectives)
        end
        
        # Early stopping
        if config.early_stop_generations > 0 && generations_without_improvement >= config.early_stop_generations
            if config.verbose
                println("Early stopping: no improvement for $(config.early_stop_generations) generations")
            end
            break
        end
    end
    
    # Extract final Pareto front
    pareto_front = [population[i] for i in 1:length(population) if population[i].rank == 1]
    
    # Find best for each objective
    best_per_objective = _find_best_per_objective(population, objectives)
    
    return NSGAIIResult(pareto_front, population, length(history), history, best_per_objective)
end

# ───────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────────────────────

function _create_initial_population(grammar::Grammar, n::Int, config::NSGAIIConfig, rng::AbstractRNG)
    population = Individual[]
    max_attempts = n * 5  # Prevent infinite loops
    attempts = 0
    
    while length(population) < n && attempts < max_attempts
        attempts += 1
        tree = generate_tree(grammar;
                             method = rand(rng, [FullMethod(), GrowMethod()]),
                             min_depth = config.min_depth,
                             max_depth = config.max_depth,
                             rng = rng)
        
        # Check max_nodes constraint
        if config.max_nodes <= 0 || count_nodes(tree) <= config.max_nodes
            push!(population, Individual(tree))
        end
    end
    
    # Fill remaining with smaller trees if we hit max_attempts
    while length(population) < n
        tree = generate_tree(grammar;
                             method = GrowMethod(),
                             min_depth = 1,
                             max_depth = min(3, config.max_depth),
                             rng = rng)
        push!(population, Individual(tree))
    end
    
    return population
end

function _evaluate_population!(population::Vector{Individual}, objectives::Vector{ObjectiveFunction}, data::Dict)
    for ind in population
        if isempty(ind.objectives)
            ind.objectives = Float64[obj.func(ind.tree, data) for obj in objectives]
        end
    end
end

function _create_offspring(population::Vector{Individual}, grammar::Grammar, 
                           config::NSGAIIConfig, rng::AbstractRNG)
    offspring = Individual[]
    max_attempts = config.population_size * 3  # Prevent infinite loops
    attempts = 0
    
    while length(offspring) < config.population_size && attempts < max_attempts
        attempts += 1
        
        # Select parents (with parsimony pressure if configured)
        parent1 = tournament_select(population, config.tournament_size; 
                                    rng=rng, 
                                    parsimony_tolerance=config.parsimony_tolerance)
        parent2 = tournament_select(population, config.tournament_size; 
                                    rng=rng,
                                    parsimony_tolerance=config.parsimony_tolerance)
        
        if rand(rng) < config.crossover_prob
            # Crossover
            child1_tree, child2_tree = crossover(parent1.tree, parent2.tree; 
                                                  rng=rng, max_depth=config.max_depth)
            
            # Check max_nodes constraint
            if config.max_nodes <= 0 || count_nodes(child1_tree) <= config.max_nodes
                push!(offspring, Individual(child1_tree))
            end
            if length(offspring) < config.population_size
                if config.max_nodes <= 0 || count_nodes(child2_tree) <= config.max_nodes
                    push!(offspring, Individual(child2_tree))
                end
            end
        else
            # Mutation
            child_tree = mutate(parent1.tree, grammar; rng=rng, max_depth=config.max_depth)
            
            # Check max_nodes constraint
            if config.max_nodes <= 0 || count_nodes(child_tree) <= config.max_nodes
                push!(offspring, Individual(child_tree))
            end
        end
    end
    
    # Fill remaining with copies of parents if we hit max_attempts
    while length(offspring) < config.population_size
        parent = tournament_select(population, config.tournament_size; 
                                   rng=rng,
                                   parsimony_tolerance=config.parsimony_tolerance)
        push!(offspring, Individual(copy_tree(parent.tree)))
    end
    
    # Occasionally simplify
    for ind in offspring
        if rand(rng) < config.simplify_prob
            ind.tree = simplify(ind.tree; grammar=grammar)
        end
    end
    
    return offspring
end

function _compute_generation_stats(population::Vector{Individual}, objectives::Vector{ObjectiveFunction}, gen::Int)
    stats = Dict{Symbol,Any}(:generation => gen)
    
    # Find best individual by first objective
    if !isempty(objectives) && !isempty(population)
        best_ind = objectives[1].minimize ? 
            argmin(ind -> ind.objectives[1], population) :
            argmax(ind -> ind.objectives[1], population)
        
        # Report best individual's values for all objectives
        for (i, obj) in enumerate(objectives)
            stats[Symbol("best_$(obj.name)")] = best_ind.objectives[i]
        end
        
        stats[:best_tree] = node_to_string(best_ind.tree)
        stats[:best_tree_size] = count_nodes(best_ind.tree)
    end
    
    # Pareto front size
    stats[:pareto_size] = count(ind -> ind.rank == 1, population)
    
    return stats
end

function _print_generation_stats(stats::Dict{Symbol,Any}, objectives::Vector{ObjectiveFunction})
    gen = stats[:generation]
    
    # Build output string
    parts = String[]
    
    for obj in objectives
        key = Symbol("best_$(obj.name)")
        if haskey(stats, key)
            push!(parts, "$(obj.name)=$(round(stats[key], digits=4))")
        end
    end
    
    push!(parts, "pareto=$(stats[:pareto_size])")
    
    if haskey(stats, :best_tree)
        tree_str = stats[:best_tree]
        if length(tree_str) > 40
            tree_str = tree_str[1:37] * "..."
        end
        push!(parts, "best=\"$tree_str\"")
    end
    
    println("  Gen $gen: ", join(parts, ", "))
end

function _find_best_per_objective(population::Vector{Individual}, objectives::Vector{ObjectiveFunction})
    best = Individual[]
    
    for (i, obj) in enumerate(objectives)
        # argmin/argmax with a function returns the element directly
        best_ind = if obj.minimize
            argmin(ind -> ind.objectives[i], population)
        else
            argmax(ind -> ind.objectives[i], population)
        end
        push!(best, best_ind)
    end
    
    return best
end

# ───────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ───────────────────────────────────────────────────────────────────────────────

"""
    curve_fitting(X::AbstractMatrix, y::AbstractVector;
                  grammar=nothing, config=NSGAIIConfig(), rng=Random.GLOBAL_RNG) -> NSGAIIResult

Convenience function for standard curve fitting (symbolic regression).
Optimizes MSE and complexity as objectives.

Note: This is just one application of symbolic optimization. For custom objectives
(aggregator discovery, scoring rules, etc.), use the `optimize` function directly.

# Arguments
- `X`: Input features matrix (rows = samples, columns = features)
- `y`: Target vector

# Keyword Arguments
- `grammar`: Grammar to use (default: arithmetic + trig)
- `var_names`: Variable names (default: [:x1, :x2, ...] based on columns)
- `config`: NSGA-II configuration
- `rng`: Random number generator
"""
function curve_fitting(X::AbstractMatrix, y::AbstractVector;
                       grammar::Union{Grammar, Nothing} = nothing,
                       var_names::Union{Vector{Symbol}, Nothing} = nothing,
                       config::NSGAIIConfig = NSGAIIConfig(),
                       rng::AbstractRNG = Random.GLOBAL_RNG)
    
    n_features = size(X, 2)
    
    # Determine variable names:
    # 1. Use explicitly provided var_names if given
    # 2. Otherwise, extract from grammar if grammar is provided
    # 3. Otherwise, use defaults [:x] or [:x1, :x2, ...]
    if var_names === nothing
        if grammar !== nothing
            # Extract variable names from the provided grammar
            var_names = [v.name for v in grammar.variables]
        else
            # Default variable names
            var_names = n_features == 1 ? [:x] : [Symbol("x$i") for i in 1:n_features]
        end
    end
    
    # Validate that we have the right number of variables
    if length(var_names) != n_features
        error("Number of variable names ($(length(var_names))) must match number of features ($n_features)")
    end
    
    # Default grammar (only if not provided)
    if grammar === nothing
        grammar = Grammar(
            binary_operators = [+, -, *, (/)],
            unary_operators = [sin, cos, exp],
            variables = var_names,
            constant_range = (-5.0, 5.0),
        )
    end
    
    # Set up objectives
    objectives = [mse_objective(), complexity_objective()]
    
    # Set up data
    data = Dict{Symbol,Any}(
        :X => X,
        :y => y,
        :var_names => var_names,
    )
    
    return optimize(grammar, objectives, data; config=config, rng=rng)
end

"""
    curve_fitting(x::AbstractVector, y::AbstractVector; kwargs...) -> NSGAIIResult

Convenience method for 1D curve fitting.
"""
function curve_fitting(x::AbstractVector, y::AbstractVector; kwargs...)
    X = reshape(x, :, 1)
    curve_fitting(X, y; kwargs...)
end

# Alias for backward compatibility
const symbolic_regression = curve_fitting
