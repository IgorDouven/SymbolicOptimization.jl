#=
Multi-Objective Optimization Example
=====================================

This example demonstrates the NSGA-II optimizer for symbolic optimization.
While we use curve fitting as a familiar example here, the same framework
applies to any domain with custom objectives.

Run this from the package directory:
    julia --project=. examples/optimization_example.jl
=#

using SymbolicOptimization
using Random

# Set seed for reproducibility
rng = Random.MersenneTwister(42)

println("="^70)
println("SymbolicOptimization.jl - Multi-Objective Optimization")
println("="^70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Problem Setup
# ─────────────────────────────────────────────────────────────────────────────

println("\n1. Problem Setup")
println("-"^50)

# Target function: f(x) = x² + 2x + 1 = (x + 1)²
target_func(x) = x^2 + 2*x + 1

# Generate training data
x_train = collect(-3.0:0.25:3.0)
y_train = target_func.(x_train)

println("Target function: f(x) = x² + 2x + 1")
println("Training points: $(length(x_train))")
println("x range: [$(minimum(x_train)), $(maximum(x_train))]")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Simple Curve Fitting
# ─────────────────────────────────────────────────────────────────────────────

println("\n2. Simple Curve Fitting (convenience function)")
println("-"^50)

# Use the convenience function
config = NSGAIIConfig(
    population_size = 100,
    max_generations = 30,
    max_depth = 6,
    verbose = true,
)

println("Running optimization...")
result = curve_fitting(x_train, y_train; config=config, rng=rng)

println("\nResults:")
println("  Generations: $(result.generations)")
println("  Pareto front size: $(length(result.pareto_front))")

# Best for MSE
best_mse = get_best(result, 1)
println("\n  Best MSE:")
println("    Expression: $(node_to_string(best_mse.tree))")
println("    MSE: $(round(best_mse.objectives[1], digits=6))")
println("    Complexity: $(Int(best_mse.objectives[2]))")

# Best for complexity
best_simple = get_best(result, 2)
println("\n  Simplest:")
println("    Expression: $(node_to_string(best_simple.tree))")
println("    MSE: $(round(best_simple.objectives[1], digits=6))")
println("    Complexity: $(Int(best_simple.objectives[2]))")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom Grammar
# ─────────────────────────────────────────────────────────────────────────────

println("\n3. Custom Grammar (arithmetic only, no trig)")
println("-"^50)

grammar = Grammar(
    binary_operators = [+, -, *, (/)],
    unary_operators = Function[],  # No unary operators
    variables = [:x],
    constant_range = (-5.0, 5.0),
)

println(grammar)

objectives = [mse_objective(), complexity_objective()]

data = Dict{Symbol,Any}(
    :X => reshape(x_train, :, 1),
    :y => y_train,
    :var_names => [:x],
)

config2 = NSGAIIConfig(
    population_size = 80,
    max_generations = 25,
    max_depth = 5,
    verbose = true,
)

println("\nRunning optimization with custom grammar...")
result2 = optimize(grammar, objectives, data; config=config2, rng=rng)

best = get_best(result2, 1)
println("\nBest result:")
println("  Expression: $(node_to_string(best.tree))")
println("  Simplified: $(node_to_string(simplify(best.tree; grammar=grammar)))")
println("  MSE: $(round(best.objectives[1], digits=6))")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pareto Front Analysis
# ─────────────────────────────────────────────────────────────────────────────

println("\n4. Pareto Front Analysis")
println("-"^50)

front = get_pareto_front(result2)
println("Pareto front ($(length(front)) solutions):")
println()

# Sort by complexity for display
sorted_front = sort(front, by=ind -> ind.objectives[2])

for (i, ind) in enumerate(sorted_front)
    local expr_str = node_to_string(ind.tree)
    if length(expr_str) > 50
        expr_str = expr_str[1:47] * "..."
    end
    println("  [$i] MSE=$(round(ind.objectives[1], digits=4)), " *
            "nodes=$(Int(ind.objectives[2])): $expr_str")
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-Dimensional Regression
# ─────────────────────────────────────────────────────────────────────────────

println("\n5. Multi-Dimensional Regression")
println("-"^50)

# Target: f(x, y) = x² + y² + xy
target_2d(x, y) = x^2 + y^2 + x*y

# Generate 2D data
using Random
Random.seed!(123)
n_points = 50
X_2d = rand(rng, n_points, 2) .* 4 .- 2  # Random points in [-2, 2]²
y_2d = [target_2d(X_2d[i,1], X_2d[i,2]) for i in 1:n_points]

println("Target function: f(x, y) = x² + y² + xy")
println("Training points: $n_points (random in [-2,2]²)")

grammar_2d = Grammar(
    binary_operators = [+, -, *, (/)],
    unary_operators = Function[],
    variables = [:x, :y],
    constant_range = (-3.0, 3.0),
)

data_2d = Dict{Symbol,Any}(
    :X => X_2d,
    :y => y_2d,
    :var_names => [:x, :y],
)

config_2d = NSGAIIConfig(
    population_size = 100,
    max_generations = 30,
    max_depth = 5,
    verbose = true,
)

println("\nRunning 2D optimization...")
result_2d = optimize(grammar_2d, objectives, data_2d; config=config_2d, rng=rng)

best_2d = get_best(result_2d, 1)
println("\nBest 2D result:")
println("  Expression: $(node_to_string(best_2d.tree))")
println("  Simplified: $(node_to_string(simplify(best_2d.tree; grammar=grammar_2d)))")
println("  MSE: $(round(best_2d.objectives[1], digits=6))")
println("  Complexity: $(Int(best_2d.objectives[2]))")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Custom Objectives
# ─────────────────────────────────────────────────────────────────────────────

println("\n6. Custom Objectives (penalizing division)")
println("-"^50)

# Custom objective: count divisions (we want fewer divisions for numerical stability)
function count_divisions(tree::AbstractNode, data::Dict)
    count = 0
    for node in flatten_tree(tree)
        if node isa FunctionNode && node.func == :/
            count += 1
        end
    end
    return Float64(count)
end

div_penalty = custom_objective(:divisions, count_divisions; minimize=true)

# Three objectives: MSE, complexity, and division count
objectives_custom = [mse_objective(), complexity_objective(), div_penalty]

config_custom = NSGAIIConfig(
    population_size = 60,
    max_generations = 15,
    max_depth = 5,
    verbose = true,
)

println("Objectives: MSE, complexity, division count")
println("\nRunning optimization...")
result_custom = optimize(grammar, objectives_custom, data; config=config_custom, rng=rng)

println("\nPareto front samples:")
front_custom = get_pareto_front(result_custom)
for (i, ind) in enumerate(front_custom[1:min(5, length(front_custom))])
    println("  [$i] MSE=$(round(ind.objectives[1], digits=4)), " *
            "nodes=$(Int(ind.objectives[2])), " *
            "divs=$(Int(ind.objectives[3]))")
    println("      $(node_to_string(ind.tree))")
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

println("\n7. Early Stopping")
println("-"^50)

config_early = NSGAIIConfig(
    population_size = 50,
    max_generations = 100,  # High limit
    early_stop_generations = 10,  # Stop if no improvement for 10 gens
    verbose = true,
)

println("Running with early stopping (stop after 10 gens without improvement)...")
result_early = optimize(grammar, objectives, data; config=config_early, rng=rng)

println("\nStopped at generation: $(result_early.generations)")

# ─────────────────────────────────────────────────────────────────────────────
println("\n" * "="^70)
println("Phase 5 complete!")
println("="^70)

println("\nNext steps (coming in future phases):")
println("  - DSL macros for easier expression building")
println("  - Pre-built grammar fragments for common domains")
println("  - Parallel evaluation support")
