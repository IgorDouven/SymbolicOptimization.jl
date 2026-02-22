#=
Evaluation Engine Example
=========================

This example demonstrates how to evaluate expression trees.

Run this from the package directory:
    julia --project=. examples/evaluation_example.jl
=#

using SymbolicOptimization

println("="^60)
println("SymbolicOptimization.jl - Evaluation Engine")
println("="^60)

# ─────────────────────────────────────────────────────────────
# 1. Basic Evaluation
# ─────────────────────────────────────────────────────────────

println("\n1. Basic Evaluation")
println("-"^50)

# Build a simple tree: (x + 1) * y
tree = FunctionNode(:*,
    FunctionNode(:+, Variable(:x), Constant(1.0)),
    Variable(:y)
)

println("Tree: ", node_to_string(tree))
println()

# Evaluate with keyword arguments
result1 = evaluate(tree, x=2.0, y=3.0)
println("evaluate(tree, x=2.0, y=3.0) = $result1")  # (2 + 1) * 3 = 9

# Evaluate with EvalContext
ctx = EvalContext(x=5.0, y=2.0)
result2 = evaluate(tree, ctx)
println("evaluate(tree, ctx) with x=5, y=2 = $result2")  # (5 + 1) * 2 = 12

# Evaluate with Dict
result3 = evaluate(tree, Dict(:x => 0.0, :y => 10.0))
println("evaluate(tree, Dict) with x=0, y=10 = $result3")  # (0 + 1) * 10 = 10

# ─────────────────────────────────────────────────────────────
# 2. Safe Evaluation
# ─────────────────────────────────────────────────────────────

println("\n2. Safe Evaluation (NaN instead of errors)")
println("-"^50)

# Division by zero
div_tree = FunctionNode(:/, Variable(:x), Variable(:y))
println("Tree: ", node_to_string(div_tree))
println("  1 / 2 = ", evaluate(div_tree, x=1.0, y=2.0))
println("  1 / 0 = ", evaluate(div_tree, x=1.0, y=0.0), " (safe: returns NaN)")

# Log of negative
log_tree = FunctionNode(:log, Variable(:x))
println("\nTree: ", node_to_string(log_tree))
println("  log(e) = ", evaluate(log_tree, x=exp(1.0)))
println("  log(-1) = ", evaluate(log_tree, x=-1.0), " (safe: returns NaN)")

# Sqrt of negative
sqrt_tree = FunctionNode(:sqrt, Variable(:x))
println("\nTree: ", node_to_string(sqrt_tree))
println("  sqrt(4) = ", evaluate(sqrt_tree, x=4.0))
println("  sqrt(-4) = ", evaluate(sqrt_tree, x=-4.0), " (safe: returns NaN)")

# ─────────────────────────────────────────────────────────────
# 3. Evaluation with Grammar
# ─────────────────────────────────────────────────────────────

println("\n3. Evaluation with Grammar")
println("-"^50)

grammar = Grammar(
    binary_operators = [+, -, *, (/)],
    unary_operators = [sin, cos, exp, log],
    variables = [:x, :y],
    constant_range = (-2.0, 2.0),
)

# sin(x) + cos(y)
trig_tree = FunctionNode(:+,
    FunctionNode(:sin, Variable(:x)),
    FunctionNode(:cos, Variable(:y))
)

println("Tree: ", node_to_string(trig_tree))
println("Grammar uses safe operators automatically")
println()

# Evaluate at various points
for (x, y) in [(0.0, 0.0), (π/2, 0.0), (0.0, π), (π/4, π/4)]
    local result = evaluate(trig_tree, grammar, EvalContext(x=x, y=y))
    println("  x=$(round(x, digits=3)), y=$(round(y, digits=3)) → $(round(result, digits=4))")
end

# ─────────────────────────────────────────────────────────────
# 4. Batch Evaluation
# ─────────────────────────────────────────────────────────────

println("\n4. Batch Evaluation")
println("-"^50)

# Quadratic: x^2 + x
quad_tree = FunctionNode(:+,
    FunctionNode(:^, Variable(:x), Constant(2.0)),
    Variable(:x)
)

println("Tree: ", node_to_string(quad_tree))
println()

# Evaluate over a dataset
data = reshape(collect(-3.0:1.0:3.0), :, 1)  # Column vector
results = evaluate_batch(quad_tree, data, [:x])

println("Batch evaluation:")
for (i, x) in enumerate(data[:, 1])
    println("  x = $x → $(results[i])")
end

# ─────────────────────────────────────────────────────────────
# 5. Compiled Evaluation (for speed)
# ─────────────────────────────────────────────────────────────

println("\n5. Compiled Evaluation")
println("-"^50)

complex_tree = FunctionNode(:+,
    FunctionNode(:*,
        FunctionNode(:sin, Variable(:x)),
        Variable(:y)
    ),
    FunctionNode(:^, Variable(:z), Constant(2.0))
)

println("Tree: ", node_to_string(complex_tree))

# Compile to a Julia function
f = compile_tree(complex_tree, [:x, :y, :z])

println("Compiled to function f(x, y, z)")
println()

# Test equivalence
println("Testing equivalence:")
for _ in 1:5
    local x, y, z = rand(3) .* 4 .- 2  # Random values in [-2, 2]
    local interpreted = evaluate(complex_tree, x=x, y=y, z=z)
    local compiled = f(x, y, z)
    local match = isapprox(interpreted, compiled; atol=1e-10) ? "✓" : "✗"
    println("  x=$(round(x,digits=2)), y=$(round(y,digits=2)), z=$(round(z,digits=2)): $match")
end

# Benchmark hint
println("\nCompiled functions are faster for repeated evaluation.")
println("Use compile_tree() when evaluating the same tree many times.")

# ─────────────────────────────────────────────────────────────
# 6. Validity Checking
# ─────────────────────────────────────────────────────────────

println("\n6. Validity Checking")
println("-"^50)

# A tree that may produce invalid results
risky_tree = FunctionNode(:log, FunctionNode(:-, Variable(:x), Constant(1.0)))
println("Tree: ", node_to_string(risky_tree), "  (log(x - 1))")
println()

test_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
println("Checking validity:")
for x in test_values
    local ctx = EvalContext(x=x)
    local valid = is_valid_on(risky_tree, ctx)
    local result = safe_evaluate(risky_tree, ctx)
    local status = valid ? "valid" : "invalid"
    println("  x = $x → $status (result = $(round(result, digits=4)))")
end

# ─────────────────────────────────────────────────────────────
# 7. Mathematical Identity Check
# ─────────────────────────────────────────────────────────────

println("\n7. Mathematical Identity: sin²(x) + cos²(x) = 1")
println("-"^50)

identity_tree = FunctionNode(:+,
    FunctionNode(:^, FunctionNode(:sin, Variable(:x)), Constant(2.0)),
    FunctionNode(:^, FunctionNode(:cos, Variable(:x)), Constant(2.0))
)

println("Tree: ", node_to_string(identity_tree))
println()

# Test at various angles
angles = [0.0, π/6, π/4, π/3, π/2, π, 3π/2, 2π]
println("Verification:")
for x in angles
    local result = evaluate(identity_tree, x=x)
    println("  x = $(round(x, digits=4)) → $(round(result, digits=10))")
end

# ─────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("Phase 3 complete! Next: Genetic operators")
println("="^60)
