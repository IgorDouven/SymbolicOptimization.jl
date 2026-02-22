#=
Genetic Operators Example
=========================

This example demonstrates the genetic programming operators:
tree generation, mutation, crossover, and simplification.

Run this from the package directory:
    julia --project=. examples/genetic_operators_example.jl
=#

using SymbolicOptimization
using Random

# Set seed for reproducibility
rng = Random.MersenneTwister(42)

println("="^60)
println("SymbolicOptimization.jl - Genetic Operators")
println("="^60)

# ─────────────────────────────────────────────────────────────
# 1. Grammar Setup
# ─────────────────────────────────────────────────────────────

println("\n1. Grammar Setup")
println("-"^50)

grammar = Grammar(
    binary_operators = [+, -, *, (/)],
    unary_operators = [sin, cos, exp],
    variables = [:x, :y],
    constant_range = (-2.0, 2.0),
)

println(grammar)

# ─────────────────────────────────────────────────────────────
# 2. Tree Generation
# ─────────────────────────────────────────────────────────────

println("\n2. Tree Generation")
println("-"^50)

# Generate with different methods
println("\nGrow method (variable depth):")
for i in 1:3
    local tree = generate_tree(grammar; method=GrowMethod(), max_depth=3, rng=rng)
    println("  Tree $i: ", node_to_string(tree), " (depth=$(tree_depth(tree)))")
end

println("\nFull method (all branches same depth):")
for i in 1:3
    local tree = generate_tree(grammar; method=FullMethod(), min_depth=2, max_depth=2, rng=rng)
    println("  Tree $i: ", node_to_string(tree), " (depth=$(tree_depth(tree)))")
end

println("\nHalf-and-half method (mixed):")
for i in 1:3
    local tree = generate_tree(grammar; method=HalfAndHalfMethod(), max_depth=3, rng=rng)
    println("  Tree $i: ", node_to_string(tree))
end

# ─────────────────────────────────────────────────────────────
# 3. Population Generation
# ─────────────────────────────────────────────────────────────

println("\n3. Population Generation")
println("-"^50)

population = generate_population(grammar, 5; max_depth=3, rng=rng)
println("Generated $(length(population)) individuals:")
for (i, tree) in enumerate(population)
    println("  [$i] ", node_to_string(tree))
end

# Unique population
unique_pop = generate_population(grammar, 5; max_depth=3, unique=true, rng=rng)
println("\nUnique population ($(length(unique_pop)) distinct trees):")
for (i, tree) in enumerate(unique_pop)
    println("  [$i] ", node_to_string(tree))
end

# ─────────────────────────────────────────────────────────────
# 4. Mutation Operators
# ─────────────────────────────────────────────────────────────

println("\n4. Mutation Operators")
println("-"^50)

# Start with a simple tree
original = FunctionNode(:+,
    FunctionNode(:*, Variable(:x), Constant(2.0)),
    Variable(:y)
)
println("Original: ", node_to_string(original))
println()

# Point mutation
println("Point mutation (swap operator/variable):")
for i in 1:3
    local mutated = mutate_point(original, grammar; rng=rng)
    println("  → ", node_to_string(mutated))
end

# Subtree mutation
println("\nSubtree mutation (replace subtree):")
for i in 1:3
    local mutated = mutate_subtree(original, grammar; rng=rng, max_depth=2)
    println("  → ", node_to_string(mutated))
end

# Hoist mutation
println("\nHoist mutation (promote subtree to root):")
for i in 1:3
    local mutated = mutate_hoist(original; rng=rng)
    println("  → ", node_to_string(mutated))
end

# Constant perturbation
tree_with_consts = FunctionNode(:+,
    FunctionNode(:*, Variable(:x), Constant(2.0)),
    Constant(1.0)
)
println("\nConstant perturbation:")
println("  Original: ", node_to_string(tree_with_consts))
for i in 1:3
    local mutated = mutate_constants(tree_with_consts; rng=rng, std=0.5)
    println("  → ", node_to_string(mutated))
end

# Insert mutation
println("\nInsert mutation (add operator above node):")
simple = Variable(:x)
println("  Original: ", node_to_string(simple))
for i in 1:3
    local mutated = mutate_insert(simple, grammar; rng=rng)
    println("  → ", node_to_string(mutated))
end

# Delete mutation
println("\nDelete mutation (remove operator):")
println("  Original: ", node_to_string(original))
for i in 1:3
    local mutated = mutate_delete(original; rng=rng)
    println("  → ", node_to_string(mutated))
end

# Combined mutation (automatic selection)
println("\nCombined mutation (random operator):")
for i in 1:5
    local mutated = mutate(original, grammar; rng=rng)
    println("  → ", node_to_string(mutated))
end

# ─────────────────────────────────────────────────────────────
# 5. Crossover Operators
# ─────────────────────────────────────────────────────────────

println("\n5. Crossover Operators")
println("-"^50)

parent1 = FunctionNode(:+,
    FunctionNode(:*, Variable(:x), Constant(2.0)),
    Variable(:y)
)
parent2 = FunctionNode(:sin,
    FunctionNode(:-, Variable(:x), Variable(:y))
)

println("Parent 1: ", node_to_string(parent1))
println("Parent 2: ", node_to_string(parent2))
println()

# Subtree crossover
println("Subtree crossover:")
for i in 1:3
    local child1, child2 = crossover_subtree(parent1, parent2; rng=rng)
    println("  Child 1: ", node_to_string(child1))
    println("  Child 2: ", node_to_string(child2))
    println()
end

# Size-fair crossover
println("Size-fair crossover (similar subtree sizes):")
for i in 1:2
    local child1, child2 = crossover_size_fair(parent1, parent2; rng=rng)
    println("  Child 1: ", node_to_string(child1))
    println("  Child 2: ", node_to_string(child2))
    println()
end

# ─────────────────────────────────────────────────────────────
# 6. Simplification
# ─────────────────────────────────────────────────────────────

println("\n6. Simplification")
println("-"^50)

# Constant folding
println("Constant folding:")
tree1 = FunctionNode(:+, Constant(1.0), Constant(2.0))
println("  $(node_to_string(tree1)) → $(node_to_string(simplify_constants(tree1)))")

tree2 = FunctionNode(:*, 
    FunctionNode(:+, Constant(1.0), Constant(2.0)),
    Variable(:x)
)
println("  $(node_to_string(tree2)) → $(node_to_string(simplify_constants(tree2)))")

# Algebraic simplification
println("\nAlgebraic simplification:")
examples = [
    FunctionNode(:+, Variable(:x), Constant(0.0)),      # x + 0
    FunctionNode(:*, Variable(:x), Constant(1.0)),      # x * 1
    FunctionNode(:*, Variable(:x), Constant(0.0)),      # x * 0
    FunctionNode(:^, Variable(:x), Constant(0.0)),      # x ^ 0
    FunctionNode(:^, Variable(:x), Constant(1.0)),      # x ^ 1
    FunctionNode(:/, Variable(:x), Constant(1.0)),      # x / 1
]

for ex in examples
    local simplified = simplify_algebra(ex)
    println("  $(node_to_string(ex)) → $(node_to_string(simplified))")
end

# Combined simplification
println("\nCombined simplification:")
complex = FunctionNode(:*,
    FunctionNode(:+, Variable(:x), Constant(0.0)),
    FunctionNode(:+, Constant(1.0), Constant(1.0))
)
println("  Original: ", node_to_string(complex))
println("  Simplified: ", node_to_string(simplify(complex; grammar=grammar)))

# ─────────────────────────────────────────────────────────────
# 7. Evolution Simulation (Simple)
# ─────────────────────────────────────────────────────────────

println("\n7. Simple Evolution Simulation")
println("-"^50)

# Target function: x^2 + x
target_func(x) = x^2 + x

# Generate data
xs = collect(-2.0:0.5:2.0)
ys = target_func.(xs)

# Fitness function (lower is better)
function fitness(tree)
    local total_error = 0.0
    for (x, y) in zip(xs, ys)
        local pred = safe_evaluate(tree, EvalContext(x=x, y=0.0))
        if isnan(pred)
            return Inf
        end
        total_error += (pred - y)^2
    end
    return total_error / length(xs)
end

# Initialize population
pop = generate_population(grammar, 20; max_depth=4, rng=rng)
println("Initial population fitness:")
fitnesses = [fitness(t) for t in pop]
best_idx = argmin(fitnesses)
println("  Best: $(round(fitnesses[best_idx], digits=4)) - $(node_to_string(pop[best_idx]))")
println("  Worst: $(round(maximum(filter(isfinite, fitnesses)), digits=4))")

# Run a few generations
println("\nEvolving...")
for gen in 1:10
    global pop  # Need to modify the global pop variable
    
    # Evaluate fitness
    local fitnesses = [fitness(t) for t in pop]
    
    # Sort by fitness
    local sorted_indices = sortperm(fitnesses)
    pop = pop[sorted_indices]
    fitnesses = fitnesses[sorted_indices]
    
    # Report best
    if gen % 2 == 0 || gen == 1
        local best_fit = fitnesses[1]
        local best_expr = node_to_string(pop[1])
        println("  Gen $gen: fitness=$(round(best_fit, digits=6)), expr=$best_expr")
    end
    
    # Selection: keep top 50%
    local n_keep = div(length(pop), 2)
    local parents = pop[1:n_keep]
    
    # Create offspring
    local offspring = AbstractNode[]
    while length(offspring) < length(pop) - n_keep
        local p1 = rand(rng, parents)
        local p2 = rand(rng, parents)
        
        if rand(rng) < 0.7  # Crossover
            local c1, c2 = crossover(p1, p2; rng=rng, max_depth=6)
            push!(offspring, c1)
            if length(offspring) < length(pop) - n_keep
                push!(offspring, c2)
            end
        else  # Mutation
            push!(offspring, mutate(p1, grammar; rng=rng, max_depth=4))
        end
    end
    
    # Simplify offspring occasionally
    offspring = [rand(rng) < 0.1 ? simplify(o; grammar=grammar) : o for o in offspring]
    
    # New population
    pop = vcat(parents, offspring)
end

# Final best
fitnesses = [fitness(t) for t in pop]
best_idx = argmin(fitnesses)
println("\nFinal best:")
println("  Fitness: $(round(fitnesses[best_idx], digits=6))")
println("  Expression: $(node_to_string(pop[best_idx]))")
println("  Simplified: $(node_to_string(simplify(pop[best_idx]; grammar=grammar)))")

# ─────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("Phase 4 complete! Next: NSGA-II optimization")
println("="^60)
