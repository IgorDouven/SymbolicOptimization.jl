#=
DSL Example: User-Friendly Symbolic Optimization
=================================================

This example demonstrates the simplified DSL interface for SymbolicOptimization.
No need to manually create Grammar objects, configs, or data dictionaries!

Run from package directory:
    julia --project=. examples/dsl_example.jl
=#

using SymbolicOptimization
using Random

println("="^70)
println("DSL Example: User-Friendly Symbolic Optimization")
println("="^70)

# ═══════════════════════════════════════════════════════════════════════════════
# Example 1: Basic Symbolic Regression (One-liner style)
# ═══════════════════════════════════════════════════════════════════════════════

println("\n1. Basic Symbolic Regression")
println("-"^50)

# Generate some data: y = x² - 1
Random.seed!(42)
X = reshape(range(-3, 3, length=31), :, 1)
y = vec(X.^2 .- 1)

println("Target function: f(x) = x² - 1")
println("Data points: $(length(y))")

# Using the fluent builder interface
prob = symbolic_problem(
    X = X,
    y = y,
    variables = [:x],
    binary_operators = [+, -, *, /],
    unary_operators = Function[],
    constants = (-2.0, 2.0),
    population = 200,
    generations = 50,
    seed = 42,
    verbose = true
)

result = solve(prob)

# Get the best solution
best_sol = best(result)
println("\n--- Result ---")
println("Best expression: $(best_sol.expression)")
println("MSE: $(round(best_sol.objectives[1], digits=6))")
println("Complexity: $(Int(best_sol.objectives[2])) nodes")

# ═══════════════════════════════════════════════════════════════════════════════
# Example 2: Step-by-step Builder Pattern
# ═══════════════════════════════════════════════════════════════════════════════

println("\n\n2. Builder Pattern (Step-by-step)")
println("-"^50)

# Generate data: y = sin(x)
X2 = reshape(range(-2π, 2π, length=50), :, 1)
y2 = sin.(vec(X2))

println("Target function: f(x) = sin(x)")

# Build problem step by step
prob2 = SymbolicProblem()
variables!(prob2, :x)
operators!(prob2, binary=[+, -, *, /], unary=[sin, cos])
data!(prob2, X=X2, y=y2)
config!(prob2, population=150, generations=40, seed=123, verbose=false)

println("Running optimization (silent mode)...")
result2 = solve(prob2)

best2 = best(result2)
println("\n--- Result ---")
println("Best expression: $(best2.expression)")
println("MSE: $(round(best2.objectives[1], digits=6))")

# Show Pareto front
println("\nPareto frontier:")
front = pareto_front(result2)
# Deduplicate
seen = Set{String}()
for sol in sort(front, by=s->s.objectives[2])
    if sol.expression ∉ seen
        push!(seen, sol.expression)
        println("  $(Int(sol.objectives[2])) nodes, MSE=$(round(sol.objectives[1], digits=6)): $(sol.expression)")
    end
    length(seen) >= 5 && break
end

# ═══════════════════════════════════════════════════════════════════════════════
# Example 3: Multi-dimensional Regression
# ═══════════════════════════════════════════════════════════════════════════════

println("\n\n3. Multi-dimensional Regression")
println("-"^50)

# Generate data: z = x*y + x - y
Random.seed!(42)
n_points = 100
X3 = 4 .* rand(n_points, 2) .- 2  # Random points in [-2, 2]²
y3 = X3[:, 1] .* X3[:, 2] .+ X3[:, 1] .- X3[:, 2]

println("Target function: f(x,y) = x*y + x - y")
println("Data points: $n_points random samples in [-2,2]²")

prob3 = symbolic_problem(
    X = X3,
    y = y3,
    variables = [:x, :y],
    binary_operators = [+, -, *, /],
    population = 200,
    generations = 50,
    max_nodes = 20,
    seed = 42,
    verbose = false
)

println("Running optimization...")
result3 = solve(prob3)

best3 = best(result3)
println("\n--- Result ---")
println("Best expression: $(best3.expression)")
println("MSE: $(round(best3.objectives[1], digits=6))")

# Evaluate on new data
X_test = [1.0 2.0; -1.0 0.5; 2.0 -1.0]
predictions = evaluate_best(result3, X_test)
actual = X_test[:, 1] .* X_test[:, 2] .+ X_test[:, 1] .- X_test[:, 2]

println("\nVerification:")
for i in 1:size(X_test, 1)
    println("  f($(X_test[i,1]), $(X_test[i,2])): predicted=$(round(predictions[i], digits=4)), actual=$(round(actual[i], digits=4))")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Example 4: Exponential/Logarithmic Functions
# ═══════════════════════════════════════════════════════════════════════════════

println("\n\n4. Exponential Functions")
println("-"^50)

# Generate data: y = exp(-x)
X4 = reshape(range(0, 3, length=40), :, 1)
y4 = exp.(-vec(X4))

println("Target function: f(x) = exp(-x)")

prob4 = symbolic_problem(
    X = X4,
    y = y4,
    variables = [:x],
    binary_operators = [+, -, *, /],
    unary_operators = [exp, log],
    constants = (-2.0, 2.0),
    population = 200,
    generations = 60,
    max_depth = 5,
    seed = 42,
    verbose = false
)

println("Running optimization...")
result4 = solve(prob4)

best4 = best(result4)
println("\n--- Result ---")
println("Best expression: $(best4.expression)")
println("MSE: $(round(best4.objectives[1], digits=6))")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("Summary: DSL Interface Features")
println("="^70)

println("""

The DSL provides two main ways to define problems:

1. ONE-LINER STYLE using `symbolic_problem()`:
   
   result = solve(symbolic_problem(
       X = data_X, y = data_y,
       variables = [:x, :y],
       binary_operators = [+, -, *, /],
       population = 200, generations = 100
   ))

2. BUILDER PATTERN for step-by-step construction:
   
   prob = SymbolicProblem()
   variables!(prob, :x, :y)
   operators!(prob, binary=[+,-,*,/], unary=[sin,cos])
   data!(prob, X=data_X, y=data_y)
   config!(prob, population=200, generations=100)
   result = solve(prob)

Both produce a `SymbolicResult` with convenient accessors:
- `best(result)` → best solution (expression + objectives)
- `pareto_front(result)` → all Pareto-optimal solutions
- `evaluate_best(result, X_new)` → evaluate on new data

No need to manually create Grammar, NSGAIIConfig, or data dictionaries!
""")

println("="^70)
println("Example complete!")
println("="^70)
