# DSL Interface

The Domain-Specific Language (DSL) provides the simplest way to use SymbolicOptimization.

## Creating Problems

Use [`symbolic_problem`](@ref) with keyword arguments:

```julia
prob = symbolic_problem(
    X = data_matrix,           # Input data (rows = samples, cols = variables)
    y = targets,               # Target values
    variables = [:x, :y, :z],  # Variable names (auto-generated if omitted)
    binary_operators = [+, -, *, /],
    unary_operators = [sin, cos, exp, log],
    constants = (-2.0, 2.0),   # Range for random constants
    objectives = [:mse, :complexity],  # What to optimize
    mode = :regression,        # :regression or :aggregation
    population = 200,
    generations = 100,
    max_depth = 6,
    max_nodes = 30,
    seed = 42,
    verbose = true
)
```

## Evaluation Modes

- **`:regression`** (default): Standard symbolic regression. Each row of X is a sample.
- **`:aggregation`**: For aggregator discovery. Variables represent forecasters (columns of X), formulas combine their predictions.

```julia
result = solve(symbolic_problem(
    X = forecaster_predictions,
    y = ground_truth,
    mode = :aggregation,
    objectives = [:brier, :complexity]
))
```

## Built-in Objectives

- `:mse` — Mean squared error
- `:brier` — Brier score (for probability predictions vs 0/1 truth)
- `:complexity` — Number of nodes in the expression tree

## Builder Pattern

For step-by-step construction, use [`SymbolicProblem`](@ref):

```julia
prob = SymbolicProblem()
variables!(prob, :x, :y)
operators!(prob, binary=[+, -, *, /], unary=[sin])
constants!(prob, (-5.0, 5.0), probability=0.3)
objectives!(prob, :mse, :complexity)
mode!(prob, :aggregation)
add_objective!(prob, :custom, my_func)
add_function!(prob, :my_op, x -> ...)
data!(prob, X=X, y=y, extra=extra_data)
config!(prob, population=200, generations=100)
```

## Solving and Results

```julia
result = solve(prob)

best_sol = best(result)
best_sol.expression                     # Formula as string
best_sol.objectives                     # [mse, complexity, ...]

front = pareto_front(result)            # All Pareto-optimal solutions
predictions = evaluate_best(result, X)  # Evaluate on new data
```

## Macro Interface

For quick one-liners:

```julia
result = @symbolic_regression(X, y, population=200, generations=50)
```

## DSL Functions

```@docs
symbolic_problem
SymbolicProblem
solve
best
pareto_front
evaluate_best
variables!
operators!
constants!
objectives!
mode!
data!
config!
add_objective!
add_function!
```
