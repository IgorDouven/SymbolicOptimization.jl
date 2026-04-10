# SymbolicOptimization.jl

A Julia package for **multi-objective symbolic optimization** using grammar-guided genetic programming and NSGA-II.

## What is Symbolic Optimization?

Symbolic optimization evolves mathematical expressions to optimize arbitrary objectives. The result is an interpretable formula, not a black-box model.

While symbolic regression finds `f(x) ≈ y` by minimizing prediction error, symbolic optimization is the general framework for searching expression space with *any* objectives — aggregator discovery, belief update rules, scoring rule design, and more.

## Quick Start

```julia
using SymbolicOptimization

# Generate some data
X = reshape(-3:0.2:3, :, 1)
y = vec(X.^2 .- 1)  # Target: x² - 1

# Define and solve
result = solve(symbolic_problem(
    X = X,
    y = y,
    variables = [:x],
    binary_operators = [+, -, *, /],
    population = 200,
    generations = 50,
))

best_sol = best(result)
println(best_sol.expression)
```

## API Levels

SymbolicOptimization offers multiple API levels:

1. **DSL (simplest)**: `symbolic_problem()` + `solve()`
2. **Builder pattern**: `SymbolicProblem()` with `variables!()`, `operators!()`, etc.
3. **JuMP-style**: `@variables`, `@operators`, `@objective`, `optimize!()`
4. **Low-level**: Manual `Grammar` construction + `optimize()`
5. **Advanced**: Custom contexts and evaluators

See the [API Reference](@ref) for full details.

## JuMP-style Example

```julia
using SymbolicOptimization

m = SymbolicModel()
@variables(m, x, y)
@operators(m, binary=[+, -, *, /], unary=[sin, cos])
@objective(m, :mse)
@objective(m, :complexity)
@data(m, X=X, y=y)
optimize!(m)
println(best(m))
```

## Deep Simplification with Symbolics.jl

Load the Symbolics.jl extension for algebraic simplification of discovered formulas:

```julia
using SymbolicOptimization
using Symbolics  # triggers the extension

tree = FunctionNode(:*,
    FunctionNode(:+, Variable(:x), Constant(1.0)),
    FunctionNode(:-, Variable(:x), Constant(1.0)))
deep_simplify(tree)  # → x² - 1
```
