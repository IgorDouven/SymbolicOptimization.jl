#=
Symbolic Regression: Discovering Mathematical Expressions
==========================================================

This example demonstrates symbolic regression - the task of discovering
mathematical expressions that fit data. Unlike neural networks or
traditional regression, symbolic regression produces interpretable,
closed-form expressions.

Key concepts demonstrated:
1. Basic curve fitting with known target functions
2. Rediscovering physical laws from data
3. Handling noisy data
4. Interpreting the Pareto frontier (accuracy vs complexity trade-off)
5. Multi-dimensional symbolic regression

Run from package directory:
    julia --project=. examples/symbolic_regression_discovery.jl
=#

using SymbolicOptimization
using Random
using Statistics
using Printf

println("="^70)
println("Symbolic Regression: Discovering Mathematical Expressions")
println("="^70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic Example: Rediscovering a Polynomial
# ─────────────────────────────────────────────────────────────────────────────

println("\n1. Basic Example: Rediscovering a Polynomial")
println("-"^50)

# Target: f(x) = x² - 1 = (x+1)(x-1)
# A simple quadratic that requires discovering x*x
target_poly(x) = x^2 - 1

# Generate clean training data
x_train = collect(-3.0:0.2:3.0)
y_train = target_poly.(x_train)

println("Target function: f(x) = x² - 1")
println("Training points: $(length(x_train))")
println("x range: [$(minimum(x_train)), $(maximum(x_train))]")

# Use a polynomial-only grammar
poly_grammar = Grammar(
    binary_operators = [+, -, *, (/)],
    unary_operators = Function[],
    variables = [:x],
    constant_range = (-3.0, 3.0),
    constant_prob = 0.3,
)

# Configure the search - use more resources for this harder problem
config = NSGAIIConfig(
    population_size = 300,
    max_generations = 80,
    max_depth = 4,
    max_nodes = 8,  # x*x - 1 is only 5 nodes
    tournament_size = 3,
    crossover_prob = 0.7,
    mutation_prob = 0.3,
    parsimony_tolerance = 0.001,
    verbose = true,
)

rng = MersenneTwister(42)

println("\nRunning symbolic regression (polynomial grammar)...")
result = symbolic_regression(x_train, y_train; grammar=poly_grammar, config=config, rng=rng)

# Analyze results
println("\n--- Results ---")
best_mse = get_best(result, 1)
best_simple = get_best(result, 2)

# Show simplified form
simplified = simplify(best_mse.tree; grammar=poly_grammar)

println("Best by MSE:")
println("  Expression: $(node_to_string(best_mse.tree))")
println("  Simplified: $(node_to_string(simplified))")
println("  MSE: $(round(best_mse.objectives[1], digits=8))")
println("  Complexity: $(Int(best_mse.objectives[2])) nodes")

# Verify on test points
println("\nVerification:")
for tx in [-2.0, -1.0, 0.0, 1.0, 2.0]
    pred = evaluate(best_mse.tree, Dict(:x => tx))
    actual = target_poly(tx)
    @printf("  f(%.1f): predicted=%.4f, actual=%.4f\n", tx, pred, actual)
end

println("\nSimplest solution:")
println("  Expression: $(node_to_string(best_simple.tree))")
println("  MSE: $(round(best_simple.objectives[1], digits=8))")
println("  Complexity: $(Int(best_simple.objectives[2])) nodes")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Physics Discovery: Kepler's Third Law
# ─────────────────────────────────────────────────────────────────────────────

println("\n\n2. Physics Discovery: Kepler's Third Law")
println("-"^50)

#=
Kepler's Third Law: T² ∝ a³
Where T = orbital period, a = semi-major axis

We'll present data as: given orbital radius r, predict orbital period T
The relationship is: T = k * r^(3/2) for some constant k

In our solar system with r in AU and T in years, k ≈ 1
=#

# Orbital data (approximate, normalized)
# r = orbital radius (AU), T = orbital period (years)
planets = [
    (r=0.39, T=0.24),   # Mercury
    (r=0.72, T=0.62),   # Venus
    (r=1.00, T=1.00),   # Earth
    (r=1.52, T=1.88),   # Mars
    (r=5.20, T=11.86),  # Jupiter
    (r=9.54, T=29.46),  # Saturn
    (r=19.2, T=84.01),  # Uranus
    (r=30.1, T=164.8),  # Neptune
]

r_data = [p.r for p in planets]
T_data = [p.T for p in planets]

println("Task: Discover the relationship between orbital radius and period")
println("Planets: $(length(planets))")
println("True relationship: T = r^(3/2)  (Kepler's Third Law)")

# Grammar focused on power laws - sqrt enables r^(3/2) = r * sqrt(r)
kepler_grammar = Grammar(
    binary_operators = [+, -, *, (/)],
    unary_operators = [sqrt],  # Only sqrt - this enables discovering r^(3/2)
    variables = [:r],
    constant_range = (0.5, 3.0),
    constant_prob = 0.2,
)

kepler_config = NSGAIIConfig(
    population_size = 200,
    max_generations = 50,
    max_depth = 4,
    max_nodes = 8,
    parsimony_tolerance = 0.001,
    verbose = true,
)

println("\nSearching for Kepler's law...")
kepler_result = symbolic_regression(r_data, T_data; 
    grammar=kepler_grammar, config=kepler_config, rng=MersenneTwister(42))

# Select the simplest solution with good MSE (avoids overfitting to small dataset)
front = get_pareto_front(kepler_result)
# Filter to solutions with MSE < 0.02 (clearly good fits)
good_solutions = filter(ind -> ind.objectives[1] < 0.02, front)
if isempty(good_solutions)
    best_kepler = get_best(kepler_result, 1)
else
    best_kepler = argmin(ind -> ind.objectives[2], good_solutions)  # Simplest among good
end

println("\n--- Discovered Law ---")
println("Expression: $(node_to_string(best_kepler.tree))")
println("MSE: $(round(best_kepler.objectives[1], digits=6))")

# Verify on test point
test_r = 2.0
predicted_T = evaluate(best_kepler.tree, Dict(:r => test_r))
true_T = test_r^1.5
println("\nVerification at r=2.0:")
println("  Predicted T: $(round(predicted_T, digits=4))")
println("  True T (r^1.5): $(round(true_T, digits=4))")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Discovering Trigonometric Functions
# ─────────────────────────────────────────────────────────────────────────────

println("\n\n3. Discovering Trigonometric Functions")
println("-"^50)

# Target: f(x) = sin(x) - a classic test for symbolic regression
target_trig(x) = sin(x)

# Generate data with minimal noise
rng_noise = MersenneTwister(999)
x_trig = collect(-2π:0.2:2π)
noise_level = 0.02
y_clean = target_trig.(x_trig)
y_trig = y_clean .+ noise_level .* randn(rng_noise, length(x_trig))

println("Target function: f(x) = sin(x)")
println("Training points: $(length(x_trig))")
println("x range: [-2π, 2π]")
println("Noise level: $(noise_level) (minimal)")

# Grammar with trigonometric functions
trig_grammar = Grammar(
    binary_operators = [+, -, *, (/)],
    unary_operators = [sin, cos],
    variables = [:x],
    constant_range = (-2.0, 2.0),
    constant_prob = 0.2,  # Fewer constants, more structure
)

trig_config = NSGAIIConfig(
    population_size = 150,
    max_generations = 40,
    max_depth = 4,
    max_nodes = 8,  # sin(x) is only 2 nodes
    parsimony_tolerance = 0.001,
    verbose = true,
)

println("\nSearching for trigonometric function...")
trig_result = symbolic_regression(x_trig, y_trig;
    grammar=trig_grammar, config=trig_config, rng=MersenneTwister(456))

# Find the simplest solution among those with the best MSE
front = get_pareto_front(trig_result)
best_mse_val = minimum(ind.objectives[1] for ind in front)
best_solutions = filter(ind -> ind.objectives[1] ≈ best_mse_val, front)
best_trig = argmin(ind -> ind.objectives[2], best_solutions)  # Simplest among best

# Compare on clean test data
x_test = collect(-2π:0.1:2π)
y_test = target_trig.(x_test)

predictions = [evaluate(best_trig.tree, Dict(:x => xi)) for xi in x_test]
test_mse = mean((predictions .- y_test).^2)

println("\n--- Results ---")
println("Discovered: $(node_to_string(best_trig.tree))")
println("Training MSE: $(round(best_trig.objectives[1], digits=6))")
println("Test MSE (clean): $(round(test_mse, digits=6))")
println("Complexity: $(Int(best_trig.objectives[2])) nodes")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pareto Frontier Analysis
# ─────────────────────────────────────────────────────────────────────────────

println("\n\n4. Pareto Frontier Analysis")
println("-"^50)

#=
The Pareto frontier shows the trade-off between accuracy and complexity.
Each point represents a solution that is not dominated by any other
(i.e., no other solution is both more accurate AND simpler).
=#

front = get_pareto_front(trig_result)

# Deduplicate by expression
seen = Set{String}()
unique_front = Individual[]
for ind in front
    expr = node_to_string(ind.tree)
    if expr ∉ seen
        push!(seen, expr)
        push!(unique_front, ind)
    end
end

# Sort by complexity
sorted_front = sort(unique_front, by=ind -> ind.objectives[2])

println("Pareto frontier ($(length(unique_front)) unique solutions):")
println("Trade-off: Lower MSE ↔ Higher Complexity")
println()
println("  Complexity │    MSE    │ Expression")
println("  ───────────┼───────────┼─" * "─"^40)

for ind in sorted_front[1:min(12, length(sorted_front))]
    expr = node_to_string(ind.tree)
    if length(expr) > 38
        expr = expr[1:35] * "..."
    end
    @printf("  %5d      │ %9.6f │ %s\n", 
            Int(ind.objectives[2]), ind.objectives[1], expr)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-Dimensional Regression
# ─────────────────────────────────────────────────────────────────────────────

println("\n\n5. Multi-Dimensional Regression")
println("-"^50)

# Target: f(x,y) = x*y + x - y  (a simple bilinear function)
target_2d(x, y) = x*y + x - y

# Generate 2D training data
rng_2d = MersenneTwister(789)
n_points = 100
X_2d = rand(rng_2d, n_points, 2) .* 4 .- 2  # Random in [-2, 2]²
y_2d = [target_2d(X_2d[i,1], X_2d[i,2]) for i in 1:n_points]

println("Target function: f(x,y) = x*y + x - y")
println("Training points: $n_points (random in [-2,2]²)")

grammar_2d = Grammar(
    binary_operators = [+, -, *, (/)],
    unary_operators = Function[],  # No unary ops needed
    variables = [:x, :y],
    constant_range = (-3.0, 3.0),
    constant_prob = 0.2,
)

config_2d = NSGAIIConfig(
    population_size = 200,
    max_generations = 50,
    max_depth = 4,
    max_nodes = 10,
    parsimony_tolerance = 0.001,
    verbose = true,
)

data_2d = Dict{Symbol,Any}(
    :X => X_2d,
    :y => y_2d,
    :var_names => [:x, :y],
)

objectives_2d = [mse_objective(), complexity_objective()]

println("\nSearching for 2D function...")
result_2d = optimize(grammar_2d, objectives_2d, data_2d; 
    config=config_2d, rng=MersenneTwister(321))

best_2d = get_best(result_2d, 1)
simplified_2d = simplify(best_2d.tree; grammar=grammar_2d)

println("\n--- Results ---")
println("Discovered: $(node_to_string(best_2d.tree))")
println("Simplified: $(node_to_string(simplified_2d))")
println("MSE: $(round(best_2d.objectives[1], digits=8))")
println("Complexity: $(Int(best_2d.objectives[2])) nodes")

# Verify on test points (avoiding y=0 since some solutions use division)
test_points = [(1.0, 2.0), (-1.0, 0.5), (2.0, -1.0)]
println("\nVerification:")
for (tx, ty) in test_points
    predicted = evaluate(best_2d.tree, Dict(:x => tx, :y => ty))
    actual = target_2d(tx, ty)
    @printf("  f(%.1f, %.1f): predicted=%.4f, actual=%.4f\n", tx, ty, predicted, actual)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Discovering Exponential Functions
# ─────────────────────────────────────────────────────────────────────────────

println("\n\n6. Discovering Exponential Functions")
println("-"^50)

#=
Exponential decay is ubiquitous in physics and biology.
Target: f(x) = exp(-x)  (simple exponential decay)
=#

target_exp(x) = exp(-x)

x_exp = collect(-2.0:0.1:3.0)
y_exp = target_exp.(x_exp)

println("Target: f(x) = exp(-x)")
println("Training points: $(length(x_exp))")

exp_grammar = Grammar(
    binary_operators = [+, -, *, (/)],
    unary_operators = [exp],
    variables = [:x],
    constant_range = (-2.0, 2.0),
    constant_prob = 0.3,
)

exp_config = NSGAIIConfig(
    population_size = 200,
    max_generations = 50,
    max_depth = 4,
    max_nodes = 8,
    parsimony_tolerance = 0.001,
    verbose = true,
)

println("\nSearching for exponential function...")
exp_result = symbolic_regression(x_exp, y_exp;
    grammar=exp_grammar, config=exp_config, rng=MersenneTwister(123))

best_exp = get_best(exp_result, 1)
println("\n--- Results ---")
println("Discovered: $(node_to_string(best_exp.tree))")
println("MSE: $(round(best_exp.objectives[1], digits=8))")
println("Complexity: $(Int(best_exp.objectives[2])) nodes")

# Verification
test_vals = [-1.0, 0.0, 1.0, 2.0]
println("\nVerification:")
for tx in test_vals
    pred = evaluate(best_exp.tree, Dict(:x => tx))
    actual = target_exp(tx)
    @printf("  f(%.1f): predicted=%.4f, actual=%.4f\n", tx, pred, actual)
end

# Show Pareto front for this problem
front_exp = get_pareto_front(exp_result)
unique_exp = unique(ind -> node_to_string(ind.tree), front_exp)
sorted_exp = sort(collect(unique_exp), by=ind -> ind.objectives[2])

println("\nPareto frontier (accuracy vs complexity):")
for ind in sorted_exp[1:min(5, length(sorted_exp))]
    @printf("  nodes=%2d, MSE=%.6f: %s\n", 
            Int(ind.objectives[2]), ind.objectives[1], 
            node_to_string(ind.tree))
end

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "="^70)
println("Summary")
println("="^70)

println("""

SYMBOLIC REGRESSION KEY POINTS:
-------------------------------
1. Produces interpretable, closed-form expressions (unlike black-box models)
2. Multi-objective: trades off accuracy vs complexity via Pareto frontier
3. Can rediscover known mathematical relationships from data
4. Works in multiple dimensions
5. Appropriate grammar design is crucial for success

EXAMPLES DEMONSTRATED:
----------------------
• Polynomial: f(x) = x² - 1
• Physics (Kepler's Law): T = r^(3/2)  
• Trigonometric: f(x) = sin(x)
• Multi-dimensional: f(x,y) = x*y + x - y
• Exponential: f(x) = exp(-x)

TIPS FOR EFFECTIVE USE:
-----------------------
• Choose grammar primitives appropriate to your domain
• Use parsimony_tolerance to prefer simpler models within accuracy threshold
• Increase population_size and max_generations for harder problems
• Examine the full Pareto frontier, not just the "best" solution
• Start with simpler grammars and add complexity as needed

SEE ALSO:
---------
• aggregator_discovery.jl - Discovering probability aggregation functions
• belief_updating_discovery.jl - Discovering belief update heuristics
""")

println("="^70)
println("Example complete!")
println("="^70)
