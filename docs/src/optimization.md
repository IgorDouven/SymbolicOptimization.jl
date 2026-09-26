# Optimization

SymbolicOptimization uses NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization. The core API is [`optimize`](@ref) — the DSL and Model API are built on top of it.

## General Optimization

```julia
using SymbolicOptimization

grammar = Grammar(
    binary_operators = [+, -, *, /],
    unary_operators = [sin, cos],
    variables = [:x, :y],
    constant_range = (-5.0, 5.0),
)

objectives = [
    custom_objective(:my_metric, (tree, data) -> compute_score(tree, data)),
    complexity_objective(),
]

data = Dict(:training_set => ..., :validation_set => ...)

config = NSGAIIConfig(population_size=100, max_generations=50)
result = optimize(grammar, objectives, data; config=config)

for ind in get_pareto_front(result)
    println("Objectives: $(ind.objectives)")
    println("Expression: $(node_to_string(ind.tree))")
end
```

## Curve Fitting

For the common case of fitting ``y \approx f(x)``:

```julia
x = collect(-2.0:0.1:2.0)
y = @. x^2 + 2x + 1

result = curve_fitting(x, y;
    config = NSGAIIConfig(population_size=100, max_generations=50)
)

best = get_best(result, 1)
println(node_to_string(best.tree))
```

## Symbolic Regression

A convenience wrapper that sets up grammar and objectives automatically:

```julia
result = symbolic_regression(X, y; config=config)
```

## Built-in Objectives

```julia
mse_objective()           # Mean squared error
mae_objective()           # Mean absolute error
complexity_objective()    # Number of nodes
depth_objective()         # Tree depth
custom_objective(name, f) # f(tree, data) -> Float64
```

## Configuration

[`NSGAIIConfig`](@ref) controls the optimization process:

```julia
config = NSGAIIConfig(
    population_size = 100,
    max_generations = 50,
    # ... additional parameters
)
```

### Deduplication

Genetic programming populations often fill up with copies of the same expression, which
reduces diversity and wastes evaluations. Setting `deduplicate = true` removes duplicates
before each environmental selection step: of all individuals whose expressions print
identically (via `node_to_string`), only the first is kept. If fewer than `population_size`
individuals remain, the population is topped up with freshly generated random individuals,
which are evaluated and themselves kept distinct where possible.

```julia
config = NSGAIIConfig(population_size = 100, max_generations = 50, deduplicate = true)
```

The option is off by default (`deduplicate = false`), so existing code behaves as before.

### Reproducibility

Pass an explicit random number generator to make a run reproducible:

```julia
using Random
result = optimize(grammar, objectives, data; config=config, rng=MersenneTwister(42))
```

All randomness during tree generation, mutation, crossover, selection and constant
sampling is drawn from `rng`, so the result does not depend on the state of the global RNG.
Custom constant samplers passed to [`ConstantSpec`](@ref) should accept an `AbstractRNG`
argument (`sampler(rng)`) to take part in this; zero-argument samplers still work but draw
from whatever RNG they use internally.

## Results

[`NSGAIIResult`](@ref) holds the optimization output:

```julia
result = optimize(grammar, objectives, data; config=config)

get_best(result, obj_index)   # Best individual for objective i
get_pareto_front(result)      # All Pareto-optimal individuals
```

Each [`Individual`](@ref) has:
- `tree` — the expression tree (`AbstractNode`)
- `objectives` — vector of objective values

## Optimization Reference

```@docs
optimize
curve_fitting
symbolic_regression
NSGAIIConfig
NSGAIIResult
Individual
ObjectiveFunction
mse_objective
mae_objective
complexity_objective
depth_objective
custom_objective
get_best
get_pareto_front
dominates
nondominated_sort!
compute_crowding_distance!
tournament_select
environmental_select!
```
