# SymbolicOptimization.jl

A Julia package for **multi-objective symbolic optimization** using grammar-guided genetic programming and NSGA-II.

## What is Symbolic Optimization?

Symbolic optimization evolves mathematical expressions to optimize arbitrary objectives. The result is an interpretable formula, not a black-box model.

**This is not (just) symbolic regression.** While symbolic regression finds ``f(x) \approx y`` by minimizing prediction error, symbolic optimization is the general framework for searching expression space with *any* objectives:

| Application | Objective(s) | Output |
|-------------|--------------|--------|
| **Aggregator discovery** | Calibration, accuracy on crowd predictions | Formula combining forecaster estimates |
| **Belief update rules** | Match normative Bayesian updates | Heuristic for updating credences |
| **Scoring rule design** | Proper scoring, discrimination | Formula evaluating forecaster quality |
| **Curve fitting** | MSE on (x, y) data | Regression formula |

## Quick Start

The simplest way to use SymbolicOptimization is via the [DSL Interface](@ref "DSL Interface"):

```julia
using SymbolicOptimization

# Generate some data
X = reshape(-3:0.2:3, :, 1)
y = vec(X.^2 .- 1)  # Target: x² - 1

# Define and solve the problem in one call
result = solve(symbolic_problem(
    X = X,
    y = y,
    variables = [:x],
    binary_operators = [+, -, *, /],
    population = 200,
    generations = 50
))

# Get the best solution
best_sol = best(result)
println(best_sol.expression)  # Something like "(x * x) - 1.0"
println(best_sol.objectives)  # [MSE, complexity]
```

For more control, see the [Model API](@ref "Model API") which provides a JuMP-inspired macro-based interface.

## Installation

```julia
using Pkg
Pkg.add("SymbolicOptimization")
```

## Package Overview

| Component | Description |
|-----------|-------------|
| [DSL Interface](@ref "DSL Interface") | High-level `symbolic_problem()` and builder pattern |
| [Model API](@ref "Model API") | JuMP-style `SymbolicModel` with macros |
| [Optimization](@ref) | NSGA-II engine, objectives, results |
| [Core Types & Trees](@ref "Core Types & Trees") | `AbstractNode`, tree utilities |
| [Grammar System](@ref "Grammar System") | Typed/untyped grammars, safe operations |
| [Evaluation](@ref) | Expression evaluation, context-aware evaluation |
| [Genetic Operators](@ref "Genetic Operators") | Generation, mutation, crossover, simplification |
| [Advanced Topics](@ref "Advanced Topics") | Symbolics.jl integration, constraints, policy problems |

## Limitations

**For high-dimensional symbolic regression** (e.g., 100+ variables), consider using
[SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) which
implements multi-population island models optimized for that use case.

SymbolicOptimization.jl excels at:
- Custom objective functions beyond MSE
- Domain-specific operators and grammars
- Context-aware evaluation (belief updating, aggregation)
- Problems with smaller search spaces but complex objectives
