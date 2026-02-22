#=
Aggregator Discovery (Simplified DSL Version)
==============================================

This example shows how to discover crowd wisdom aggregators using the 
simplified DSL interface. Compare with aggregator_discovery.jl for the
full low-level version with more detailed analysis.

The goal: Find formulas that combine probability forecasts from multiple
people to produce better calibrated and more accurate aggregate predictions.

Run from package directory:
    julia --project=. examples/aggregator_discovery_simple.jl
=#

using SymbolicOptimization
using Statistics
using Random

println("="^70)
println("Aggregator Discovery (DSL Version)")
println("="^70)

# ═══════════════════════════════════════════════════════════════════════════════
# Generate Synthetic Crowd Data
# ═══════════════════════════════════════════════════════════════════════════════

println("\n1. Generating Synthetic Crowd Data")
println("-"^50)

Random.seed!(42)

# Parameters
n_claims = 200
n_forecasters = 5

# Ground truth (binary outcomes)
truth = rand(Bool, n_claims)

# Generate forecaster predictions with varying skill levels
# Better forecasters have predictions closer to truth
skill_levels = [0.85, 0.75, 0.65, 0.60, 0.55]  # p1 is best, p5 is worst

predictions = zeros(n_claims, n_forecasters)
for j in 1:n_forecasters
    skill = skill_levels[j]
    for i in 1:n_claims
        if truth[i]
            predictions[i, j] = skill + (1 - skill) * rand() * 0.3
        else
            predictions[i, j] = (1 - skill) * (0.3 + 0.7 * rand())
        end
        predictions[i, j] = clamp(predictions[i, j], 0.01, 0.99)
    end
end

# Individual Brier scores
for j in 1:n_forecasters
    brier = mean((predictions[:, j] .- truth).^2)
    println("  Forecaster p$j: Brier = $(round(brier, digits=4))")
end

# Simple average baseline
avg_pred = mean(predictions, dims=2)
avg_brier = mean((vec(avg_pred) .- truth).^2)
println("\n  Simple average: Brier = $(round(avg_brier, digits=4))")

# ═══════════════════════════════════════════════════════════════════════════════
# Run Optimization with DSL
# ═══════════════════════════════════════════════════════════════════════════════

println("\n2. Running Symbolic Optimization")
println("-"^50)

# Build the problem using the DSL - now with aggregation mode!
# Variables p1-p5 represent each forecaster's prediction
# The formula combines them to minimize Brier score

result = solve(symbolic_problem(
    X = predictions,              # rows = claims, cols = forecasters
    y = Float64.(truth),          # ground truth (0/1)
    mode = :aggregation,          # aggregation mode (not regression)
    objectives = [:brier, :complexity],  # minimize Brier score and complexity
    constants = (0.0, 2.0),       # weights in [0, 2]
    population = 200,
    generations = 80,
    max_depth = 5,
    max_nodes = 20,
    seed = 42,
    verbose = true
))

# ═══════════════════════════════════════════════════════════════════════════════
# Analyze Results
# ═══════════════════════════════════════════════════════════════════════════════

println("\n\n3. Results")
println("-"^50)

# Best solution
best_sol = best(result)
println("Best discovered aggregator:")
println("  $(best_sol.expression)")
println("  Brier score: $(round(best_sol.objectives[1], digits=4))")
println("  Complexity: $(Int(best_sol.objectives[2])) nodes")

# Compare to baselines
println("\nBrier score comparison:")
println("  Simple average:     $(round(avg_brier, digits=4))")
println("  Discovered formula: $(round(best_sol.objectives[1], digits=4))")
improvement = 100 * (avg_brier - best_sol.objectives[1]) / avg_brier
println("  Improvement: $(round(improvement, digits=1))%")

# Show Pareto front
println("\n\nPareto Frontier (Brier vs simplicity trade-off):")
println("Complexity │  Brier   │ Expression")
println("───────────┼──────────┼─" * "─"^40)

front = pareto_front(result)
seen = Set{String}()
for sol in sort(front, by=s->s.objectives[2])
    if sol.expression ∉ seen
        push!(seen, sol.expression)
        expr = length(sol.expression) > 40 ? sol.expression[1:37] * "..." : sol.expression
        println("    $(Int(sol.objectives[2]))      │ $(round(sol.objectives[1], digits=6)) │ $expr")
    end
    length(seen) >= 8 && break
end

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("Summary")
println("="^70)

println("""

This example demonstrates the DSL interface for aggregator discovery.

KEY DSL FEATURES FOR AGGREGATION:
- mode = :aggregation tells the DSL that variables are forecaster predictions
- objectives = [:brier, :complexity] uses built-in Brier score
- Variables are auto-named p1, p2, ... for each forecaster column

COMPARED TO THE FULL VERSION (aggregator_discovery.jl):
- No manual Grammar construction
- No NSGAIIConfig setup  
- No custom objective function definition
- No data dictionary management
- ~30 lines vs ~500 lines

ONE-LINER VERSION:
```julia
result = solve(symbolic_problem(
    X = predictions,
    y = Float64.(truth),
    mode = :aggregation,
    objectives = [:brier, :complexity]
))
```

For more sophisticated aggregator discovery with custom evaluation
contexts and domain-specific operators, see aggregator_discovery.jl.
""")

println("="^70)
println("Example complete!")
println("="^70)
