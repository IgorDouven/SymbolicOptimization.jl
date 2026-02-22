#=
JuMP-like API Examples for SymbolicOptimization
================================================

This file demonstrates the new declarative API inspired by JuMP.jl.
Three levels of usage are shown:
  1. Step-by-step with macros  (most explicit)
  2. Functional API            (no macros, plain functions)
  3. @problem block            (most concise)

Run from package directory:
    julia --project=. examples/api_example.jl
=#

using SymbolicOptimization
using Random
using Statistics

println("="^70)
println("JuMP-like API for SymbolicOptimization")
println("="^70)

# ═══════════════════════════════════════════════════════════════════════════════
# Example 1: Symbolic Regression with Macros
# ═══════════════════════════════════════════════════════════════════════════════

println("\n1. Symbolic Regression (macro API)")
println("-"^50)

# Generate data: y = x² + 2x - 1
Random.seed!(42)
X_data = reshape(range(-3, 3, length=50), :, 1)
y_data = vec(X_data.^2 .+ 2 .* X_data .- 1)
println("Target: f(x) = x² + 2x - 1")

# ── Define the model ──
m = SymbolicModel()

@variables(m, x)
@operators(m, binary=[+, -, *, /], unary=[sin, cos])
@constants(m, -3.0..3.0, probability=0.3)

@objective(m, Min, :mse)
@objective(m, Min, :complexity)

@data(m, X=X_data, y=y_data)
@config(m, population=200, generations=50, seed=42, verbose=false)

# Inspect the model before solving
println("\nModel summary:")
println(m)

# ── Solve ──
println("\nOptimizing...")
optimize!(m)

# ── Results ──
b = best(m)
println("\nBest expression: $(b.expression)")
println("MSE: $(round(b.objectives[1], digits=6))")
println("Complexity: $(Int(b.objectives[2])) nodes")

# Pareto front
println("\nPareto front (top 5 by complexity):")
front = pareto_front(m)
seen = Set{String}()
for sol in sort(front, by=s -> s.objectives[2])
    if sol.expression ∉ seen
        push!(seen, sol.expression)
        println("  $(Int(sol.objectives[2])) nodes, MSE=$(round(sol.objectives[1], digits=6)): $(sol.expression)")
    end
    length(seen) >= 5 && break
end

# Predict on new data
X_test = reshape([0.0, 1.0, 2.0, -1.0], :, 1)
preds = predict(m, X_test)
actual = vec(X_test.^2 .+ 2 .* X_test .- 1)
println("\nPredictions vs actual:")
for i in 1:length(preds)
    println("  x=$(X_test[i,1]): pred=$(round(preds[i], digits=4)), actual=$(round(actual[i], digits=4))")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Example 2: Functional API (no macros)
# ═══════════════════════════════════════════════════════════════════════════════

println("\n\n2. Multidimensional Regression (functional API)")
println("-"^50)

# Generate data: z = x*y + x - y
Random.seed!(42)
n_pts = 100
X2 = 4 .* rand(n_pts, 2) .- 2
y2 = X2[:, 1] .* X2[:, 2] .+ X2[:, 1] .- X2[:, 2]
println("Target: f(x,y) = x*y + x - y")

# Build model with functional API (no macros at all)
m2 = SymbolicModel(seed=42, verbose=false)
set_variables!(m2, :x, :y)
set_operators!(m2, binary=[+, -, *, /])
set_constants!(m2, -2.0, 2.0)
add_objective!(m2, :Min, :mse)
add_objective!(m2, :Min, :complexity)
set_data!(m2, X=X2, y=y2)
set_config!(m2, population=200, generations=50, max_nodes=20)

println("Optimizing...")
optimize!(m2)

b2 = best(m2)
println("\nBest expression: $(b2.expression)")
println("MSE: $(round(b2.objectives[1], digits=6))")

# ═══════════════════════════════════════════════════════════════════════════════
# Example 3: Confirmation Measure Discovery with Seeds
# ═══════════════════════════════════════════════════════════════════════════════

println("\n\n3. Confirmation Measure Discovery (seeds + custom objectives)")
println("-"^50)

# This demonstrates the key use case: discovering symbolic formulas
# for confirmation measures in Bayesian epistemology.

# For this demo, we use a simplified evaluator
# (the real version would use generate_trial from the full example)

# Create some demo trial data
Random.seed!(42)
n_trials = 200
demo_trials = [
    Dict{Symbol,Any}(
        :pH => rand(), :pE => rand(),
        :pH_E => rand(), :pE_H => rand(),
        :pE_notH => rand(), :pnotH => 1.0 - rand() * 0.8,
        :pH_notE => rand()
    )
    for _ in 1:n_trials
]
demo_labels = rand(Bool, n_trials)

m3 = SymbolicModel()

# Declare all probability variables
@variables(m3, pH, pE, pH_E, pE_H, pE_notH, pnotH, pH_notE)

# Operators: arithmetic + log for logarithmic measures
@operators(m3, binary=[+, -, *, /], unary=[log])

# No random constants for confirmation measures
@constants(m3, nothing)

# Custom objective: AUC-based truth-tracking + complexity penalty
@objective(m3, Min, :auc_penalty, (tree, env) -> begin
    trials = env[:trials]
    labels = env[:labels]
    
    scores = Float64[]
    for trial in trials
        val = evaluate(tree, trial)
        push!(scores, isfinite(val) ? val : 0.0)
    end
    
    # Skip degenerate solutions
    if std(scores) < 1e-10
        return 1.0
    end
    
    auc = compute_auc(scores, labels)
    complexity_penalty = 0.002 * count_nodes(tree)
    return 1.0 - auc + complexity_penalty
end)

@objective(m3, Min, :complexity)

# ── Seed with known confirmation measures ──
# These are written in natural math syntax – no manual tree building!
@seed(m3, pH_E - pH)                    # Difference measure (d)
@seed(m3, pH_E - pH_notE)              # Christensen's s
@seed(m3, log(pE_H / pE_notH))        # Good's log-likelihood ratio (l)
@seed(m3, (pH_E - pH) / (1.0 - pH))   # Normalized difference

# Data
@data(m3, trials=demo_trials, labels=demo_labels)

# Configuration
@config(m3, population=300, generations=80, max_depth=7, max_nodes=30, seed=42, verbose=true)

println("Model:")
println(m3)

println("\nOptimizing...")
optimize!(m3)

b3 = best(m3)
println("\nBest confirmation measure: $(b3.expression)")
println("LaTeX: $(b3.latex)")
println("Objectives: AUC penalty = $(round(b3.objectives[1], digits=6)), complexity = $(Int(b3.objectives[2]))")

println("\nPareto front:")
front3 = pareto_front(m3)
seen3 = Set{String}()
for sol in sort(front3, by=s -> s.objectives[1])
    if sol.expression ∉ seen3
        push!(seen3, sol.expression)
        println("  $(sol.expression)")
        println("    AUC penalty=$(round(sol.objectives[1], digits=4)), nodes=$(Int(sol.objectives[2]))")
    end
    length(seen3) >= 5 && break
end

# ═══════════════════════════════════════════════════════════════════════════════
# Example 4: @problem block (most concise)
# ═══════════════════════════════════════════════════════════════════════════════

println("\n\n4. Concise @problem block")
println("-"^50)

# Generate data
Random.seed!(99)
X4 = reshape(range(0, 2π, length=40), :, 1)
y4 = sin.(vec(X4))
println("Target: f(x) = sin(x)")

m4 = @problem begin
    @variables x
    @operators binary=[+, -, *, /] unary=[sin, cos]
    @constants -2.0..2.0
    @objective Min :mse
    @objective Min :complexity
    @data X=X4 y=y4
    @config population=150 generations=40 seed=99 verbose=false
end

println("Optimizing...")
optimize!(m4)

b4 = best(m4)
println("Best: $(b4.expression)")
println("MSE: $(round(b4.objectives[1], digits=6))")

# ═══════════════════════════════════════════════════════════════════════════════
# Example 5: Aggregator Discovery
# ═══════════════════════════════════════════════════════════════════════════════

println("\n\n5. Aggregator Discovery")
println("-"^50)

# Synthetic forecaster data
Random.seed!(42)
n_claims = 200
n_forecasters = 4
truth5 = rand(Bool, n_claims)

skill_levels = [0.85, 0.75, 0.65, 0.55]
forecasts = zeros(n_claims, n_forecasters)
for j in 1:n_forecasters
    skill = skill_levels[j]
    for i in 1:n_claims
        forecasts[i, j] = truth5[i] ? skill + (1 - skill) * rand() * 0.3 :
                                       (1 - skill) * (0.3 + 0.7 * rand())
        forecasts[i, j] = clamp(forecasts[i, j], 0.01, 0.99)
    end
end

# Baseline: simple average
avg_brier = mean((mean(forecasts, dims=2) .- truth5).^2)
println("Simple average Brier: $(round(avg_brier, digits=4))")

m5 = SymbolicModel()

@variables(m5, p1, p2, p3, p4)
@operators(m5, binary=[+, -, *, /])
@constants(m5, 0.0..2.0, probability=0.3)

# Custom Brier score objective for aggregation
@objective(m5, Min, :brier, (tree, env) -> begin
    X = env[:X]
    y = env[:y]
    var_names = env[:var_names]
    
    brier_sum = 0.0
    for i in axes(X, 1)
        ctx = Dict(var_names[j] => X[i, j] for j in eachindex(var_names))
        pred = evaluate(tree, ctx)
        if isfinite(pred)
            pred_clamped = clamp(pred, 0.0, 1.0)
            brier_sum += (pred_clamped - y[i])^2
        else
            brier_sum += 1.0
        end
    end
    return brier_sum / size(X, 1)
end)

@objective(m5, Min, :complexity)

# Seed with simple average: (p1 + p2 + p3 + p4) / 4
# Note: we could also use the @seed macro but this formula needs a constant
add_seed!(m5, FunctionNode(:/, [
    FunctionNode(:+, [
        FunctionNode(:+, [Variable(:p1), Variable(:p2)]),
        FunctionNode(:+, [Variable(:p3), Variable(:p4)])
    ]),
    Constant(4.0)
]))

@data(m5, X=forecasts, y=Float64.(truth5))
@config(m5, population=200, generations=60, seed=42, verbose=false)

println("Optimizing...")
optimize!(m5)

b5 = best(m5)
println("Best aggregator: $(b5.expression)")
println("Brier score: $(round(b5.objectives[1], digits=4))")
println("Improvement over simple average: $(round(avg_brier - b5.objectives[1], digits=4))")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("API Summary")
println("="^70)

println("""

Three ways to use the package:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MACRO API (JuMP-style):

   m = SymbolicModel()
   @variables(m, x, y)
   @operators(m, binary=[+, -, *, /])
   @objective(m, Min, :mse)
   @objective(m, Min, :complexity)
   @data(m, X=my_X, y=my_y)
   @config(m, population=200, generations=100, seed=42)
   optimize!(m)
   best(m)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. FUNCTIONAL API (no macros):

   m = SymbolicModel(seed=42)
   set_variables!(m, :x, :y)
   set_operators!(m, binary=[+, -, *, /])
   add_objective!(m, :Min, :mse)
   add_objective!(m, :Min, :complexity)
   set_data!(m, X=my_X, y=my_y)
   set_config!(m, population=200, generations=100)
   optimize!(m)
   best(m)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. @problem BLOCK (most concise):

   m = @problem begin
       @variables x, y
       @operators binary=[+, -, *, /]
       @objective Min :mse
       @objective Min :complexity
       @data X=my_X y=my_y
       @config population=200 generations=100 seed=42
   end
   optimize!(m)
   best(m)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key features:
  • @seed(m, pH_E - pH)     — write math formulas, get expression trees
  • @objective(m, Min, :name, func)  — custom objectives
  • @constraint(m, :name, func)      — theoretical constraints
  • best(m), pareto_front(m)         — access results
  • predict(m, X_new)                — evaluate on new data
  • expression_latex(m)              — LaTeX output
""")

println("Done!")
