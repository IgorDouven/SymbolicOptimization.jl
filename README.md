# SymbolicOptimization.jl

[![CI](https://github.com/IgorDouven/SymbolicOptimization.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/IgorDouven/SymbolicOptimization.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/IgorDouven/SymbolicOptimization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/IgorDouven/SymbolicOptimization.jl)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://IgorDouven.github.io/SymbolicOptimization.jl/stable/)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://IgorDouven.github.io/SymbolicOptimization.jl/dev/)

A Julia package for **multi-objective symbolic optimization** using grammar-guided genetic programming and NSGA-II.

## What is Symbolic Optimization?

Symbolic optimization evolves mathematical expressions to optimize arbitrary objectives. The result is an interpretable formula, not a black-box model.

**This is not symbolic regression.** While symbolic regression finds f(x) ≈ y by minimizing prediction error, symbolic optimization is the general framework for searching expression space with *any* objectives:

| Application | Objective(s) | Output |
|-------------|--------------|--------|
| **Aggregator discovery** | Calibration, accuracy on crowd predictions | Formula combining forecaster estimates |
| **Belief update rules** | Match normative Bayesian updates | Heuristic for updating credences |
| **Scoring rule design** | Proper scoring, discrimination | Formula evaluating forecaster quality |
| **Curve fitting** | MSE on (x, y) data | Regression formula |

## Quick Start (DSL Interface)

The simplest way to use SymbolicOptimization is via the DSL (Domain-Specific Language):

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

### Builder Pattern (Step-by-Step)

For more control, use the builder pattern:

```julia
prob = SymbolicProblem()
variables!(prob, :x, :y)
operators!(prob, binary=[+, -, *, /], unary=[sin, cos, exp])
data!(prob, X=my_data, y=my_targets)
config!(prob, population=300, generations=100, seed=42)

result = solve(prob)

# Explore the Pareto frontier (trade-off between accuracy and simplicity)
for sol in pareto_front(result)
    println("$(sol.objectives) => $(sol.expression)")
end

# Evaluate the best solution on new data
predictions = evaluate_best(result, X_new)
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/IgorDouven/SymbolicOptimization.jl")
```

## Deep Algebraic Simplification (via Symbolics.jl)

The built-in `simplify` function handles basic identities (`x + 0 → x`, `x * 1 → x`, etc.) and is applied probabilistically during evolution (controlled by `simplify_prob`, default 10%). For deeper algebraic simplification of final results — like `(x + 1) * (x - 1) → x² - 1` — you can load `Symbolics.jl`:

```julia
using SymbolicOptimization
using Symbolics  # activates the extension

# After an optimization run:
b = best(result)

# Get a deeply simplified tree (still an AbstractNode, can be evaluated)
simple_tree = deep_simplify(b.tree)

# Or get simplified string / LaTeX directly
simplified_string(b.tree)   # "x^2 - 1"
simplified_latex(b.tree)    # "x^{2} - 1"
```

Safe operators (`safe_div`, `safe_log`, etc.) are automatically mapped to their standard equivalents before simplification. This is designed for presenting final results — for evaluation inside the GP loop, the original tree with safe operators should be used.

### Piecewise Formulas

If your grammar includes an indicator function like `step_func` (which returns 1 when its argument is non-negative and 0 otherwise), the evolutionary search may discover piecewise formulas of the form `step_func(cond) * A + (1 - step_func(cond)) * B`. Calling `deep_simplify` on such trees gives poor results because `step_func` is opaque to Symbolics.jl.

Use `simplify_piecewise` instead, which separates the two branches, simplifies each independently, and presents the result clearly:

```julia
r = simplify_piecewise(b.tree)
println(r)
# Piecewise formula:
#   When (pH_E - pH_notE) ≥ 0:
#     ...simplified if-branch...
#   When (pH_E - pH_notE) < 0:
#     ...simplified else-branch...

# Access individual parts:
r.if_string      # simplified if-branch as string
r.else_string    # simplified else-branch as string
r.if_latex       # LaTeX for the if-branch
r.else_latex     # LaTeX for the else-branch
r.if_branch      # simplified if-branch as AbstractNode
r.else_branch    # simplified else-branch as AbstractNode
```

### Domain Substitutions

Symbolics.jl only knows about algebraic identities, not domain-specific constraints. For instance, in probability theory `P(¬H|¬E) + P(H|¬E) = 1`, but Symbolics has no way to know this. You can declare such identities via the `substitutions` argument:

```julia
r = simplify_piecewise(b.tree, substitutions = complement_vars(
    :pnotH_notE => :pH_notE,   # P(¬H|¬E) = 1 - P(H|¬E)
    :pnotH_E    => :pH_E,      # P(¬H|E)  = 1 - P(H|E)
))
```

The `complement_vars` helper builds a substitution dictionary where each pair `a => b` means `a = 1 - b`. For more general substitutions, pass a `Dict{Symbol, AbstractNode}` directly.

### Limitations

- **Factoring**: Symbolics.jl tends to expand products rather than factor polynomials. A numerator like `pH_E * pH_notE - pH_notE^2` won't be automatically factored to `pH_notE * (pH_E - pH_notE)`. This last step typically requires human interpretation.
- **Piecewise detection**: `simplify_piecewise` assumes all `step_func(...)` calls in the tree share the same condition. Trees with multiple independent conditions are not yet supported.
- **Performance**: Deep simplification invokes the full Symbolics.jl CAS engine. It should only be used on final results, not inside the GP loop.

If you don't load `Symbolics`, none of these functions are available, and the package has no heavy dependencies.

## Overview

SymbolicOptimization.jl is designed for discovering interpretable mathematical expressions that optimize user-defined objectives.

**Key Features:**
- **User-friendly DSL**: Define problems with `symbolic_problem()` — no manual Grammar or Config setup
- **Flexible grammar specification**: DSL, programmatic API, or composable fragments
- **Multi-objective optimization**: NSGA-II finds Pareto-optimal trade-offs (accuracy vs. simplicity)
- **Custom operators**: Add domain-specific functions to the search space
- **Context-aware evaluation**: For stateful problems like belief updating

## DSL Reference

### Creating Problems

```julia
# One-liner with keyword arguments
prob = symbolic_problem(
    X = data_matrix,           # Input data (rows = samples, cols = variables)
    y = targets,               # Target values
    variables = [:x, :y, :z],  # Variable names (auto-generated if omitted)
    binary_operators = [+, -, *, /],
    unary_operators = [sin, cos, exp, log],
    constants = (-2.0, 2.0),   # Range for random constants
    objectives = [:mse, :complexity],  # What to optimize
    mode = :regression,        # :regression or :aggregation
    population = 200,          # Population size
    generations = 100,         # Number of generations
    max_depth = 6,             # Max tree depth
    max_nodes = 30,            # Max nodes per tree
    seed = 42,                 # Random seed (optional)
    verbose = true             # Print progress
)
```

### Evaluation Modes

- **`:regression`** (default): Standard symbolic regression. Each row of X is a sample.
- **`:aggregation`**: For aggregator discovery. Variables represent forecasters (columns of X),
  formulas combine their predictions. Use with `:brier` objective.

```julia
# Aggregator discovery example
result = solve(symbolic_problem(
    X = forecaster_predictions,  # rows = claims, cols = forecasters  
    y = ground_truth,            # 0/1 outcomes
    mode = :aggregation,
    objectives = [:brier, :complexity]
))
```

### Built-in Objectives

- `:mse` - Mean squared error
- `:brier` - Brier score (for probability predictions vs 0/1 truth)
- `:complexity` - Number of nodes in the expression tree

### Builder Pattern

```julia
prob = SymbolicProblem()
variables!(prob, :x, :y)                          # Set variable names
operators!(prob, binary=[+,-,*,/], unary=[sin])   # Set operators
constants!(prob, (-5.0, 5.0), probability=0.3)    # Constant range & frequency
objectives!(prob, :mse, :complexity)              # What to minimize
mode!(prob, :aggregation)                         # Set evaluation mode
add_objective!(prob, :custom, my_func)            # Add custom objective
add_function!(prob, :my_op, x -> ...)             # Add custom operator
data!(prob, X=X, y=y, extra=extra_data)           # Provide data
config!(prob, population=200, generations=100)    # Set config
```

### Solving and Results

```julia
result = solve(prob)                    # Run optimization

best_sol = best(result)                 # Best solution
best_sol.expression                     # Formula as string
best_sol.objectives                     # [mse, complexity, ...]

front = pareto_front(result)            # All Pareto-optimal solutions
predictions = evaluate_best(result, X)  # Evaluate on new data
```

## Policy Problems (Discrimination, Sequential, Custom)

For problems where objectives can't be computed point-by-point (like regression),
use `policy_problem` with a custom evaluator.

### Discrimination Problems (AUC-based)

Find formulas whose scores discriminate between positive/negative cases:

```julia
# Define a simulator that generates (inputs, label) pairs
function my_simulator(rng)
    # Generate one trial
    inputs = Dict(:x => rand(rng), :y => rand(rng), ...)
    label = some_ground_truth
    return (inputs, label)
end

result = solve(policy_problem(
    variables = [:x, :y, :z],
    evaluator = discrimination_evaluator(
        simulator = my_simulator,
        n_simulations = 1000,
        objectives = [:auc, :complexity]
    ),
    n_objectives = 2,
    population = 200,
    generations = 100
))
```

### Sequential Problems

For problems with state accumulation (e.g., belief updating):

```julia
# Sequences of steps, each with inputs and target
sequences = [
    [Dict(:prior => 0.5, :likelihood => 0.8, :target => 0.67), ...],
    ...
]

result = solve(policy_problem(
    variables = [:prior, :likelihood],
    evaluator = sequential_evaluator(
        sequences = sequences,
        target_key = :target,
        objectives = [:mse, :complexity]
    ),
    ...
))
```

### Custom Evaluators

For full control, provide your own evaluation function:

```julia
my_evaluator = (tree, env, evaluate_fn, count_nodes_fn) -> begin
    # Your evaluation logic here
    # Must return Vector{Float64} of objective values
    score = ...
    complexity = count_nodes_fn(tree)
    return [score, Float64(complexity)]
end

result = solve(policy_problem(
    variables = [...],
    evaluator = my_evaluator,
    n_objectives = 2,
    ...
))
```

## Core API

### Low-Level API (Full Control)

For advanced use cases, you can use the low-level API directly:

```julia
using SymbolicOptimization

# Build an expression tree manually
x = Variable(:x)
tree = FunctionNode(:+, 
    FunctionNode(:sin, x),
    Constant(1.0)
)

# Inspect the tree
println(node_to_string(tree))  # "(sin(x) + 1)"
println(node_to_latex(tree))   # "\left(\sin\left(x\right) + 1\right)"
println(count_nodes(tree))     # 4
println(tree_depth(tree))      # 3

# Print tree structure
print_tree(tree)
# +
#   sin
#     x
#   1
```

## Grammar System

SymbolicOptimization.jl provides a flexible grammar system inspired by (but more flexible than) SymbolicRegression.jl.

### Simple Grammar (Untyped)

For standard symbolic regression:

```julia
grammar = Grammar(
    binary_operators = [+, -, *, /],
    unary_operators = [sin, cos, exp, log],
    variables = [:x, :y],
    constant_range = (-2.0, 2.0),
)
```

### Typed Grammar

For domains with multiple types (e.g., vector/scalar operations):

```julia
grammar = Grammar(
    types = [:Scalar, :Vector],
    
    variables = [
        :ps => :Vector,    # Vector of probabilities
        :n => :Scalar,     # Scalar count
    ],
    
    operators = [
        # Reductions: Vector → Scalar
        (mean, [:Vector] => :Scalar),
        (sum, [:Vector] => :Scalar),
        (:std, [:Vector] => :Scalar),
        
        # Scalar arithmetic
        (+, [:Scalar, :Scalar] => :Scalar),
        (*, [:Scalar, :Scalar] => :Scalar),
        
        # Element-wise: (Vector, Scalar) → Vector
        (Symbol(".^"), [:Vector, :Scalar] => :Vector),
    ],
    
    constant_types = [:Scalar],
    output_type = :Scalar,
)
```

### Querying the Grammar

```julia
# Get operators by arity
unary_ops = unary_operators(grammar)
binary_ops = binary_operators(grammar)

# Get operators that produce a specific type
scalar_producers = operators_producing(grammar, :Scalar)

# Sample constants
c = sample_constant(grammar)

# Validate a tree against the grammar
valid, msg = check_tree_validity(tree, grammar)
```

### Safe Operations

All standard operators have "safe" versions that return `NaN` instead of throwing errors:

```julia
safe_div(1.0, 0.0)    # NaN (not an error)
safe_log(-1.0)        # NaN
safe_sqrt(-1.0)       # NaN
safe_pow(-2.0, 0.5)   # NaN (sqrt of negative)
```

## Evaluation Engine

Evaluate expression trees with variable bindings:

```julia
tree = FunctionNode(:+, Variable(:x), Constant(1.0))

# Multiple calling conventions
evaluate(tree, x=5.0)                           # 6.0
evaluate(tree, EvalContext(x=5.0))              # 6.0
evaluate(tree, Dict(:x => 5.0))                 # 6.0
evaluate(tree, grammar, EvalContext(x=5.0))     # Uses grammar's safe operators

# Batch evaluation over datasets
data = [1.0; 2.0; 3.0;;]  # 3×1 matrix
results = evaluate_batch(tree, data, [:x])      # [2.0, 3.0, 4.0]

# Safe evaluation (returns NaN on any error)
safe_evaluate(tree, EvalContext(x=5.0))         # 6.0
is_valid_on(tree, EvalContext(x=5.0))           # true

# Compile for speed (when evaluating same tree many times)
f = compile_tree(tree, [:x])
f(5.0)                                          # 6.0
```

### Context-Aware Evaluation

For problems where operators need access to external state (like belief updating, 
time-series processing, or domain-specific operations), use context-aware evaluation:

```julia
# Define custom operators that receive the full context
operators = Dict{Symbol, Function}(
    # add_bonus: boost item closest to data mean
    :add_bonus => (v, bonus, ctx) -> begin
        dm = ctx[:data_mean]  # Access context variables
        idx = argmin(abs.(v .- dm))
        result = copy(v)
        result[idx] += bonus
        return result
    end,
    
    # choose_by_obs: conditional on observation
    :choose_by_obs => (a, b, ctx) -> ctx[:is_heads] ? a : b
)

# Create context with variable bindings AND custom operators
ctx = EvalContext(
    Dict{Symbol, Any}(:probs => probs, :data_mean => 0.6, :is_heads => true),
    operators
)

# Evaluate - custom operators automatically receive context as last argument
tree = FunctionNode(:add_bonus, [Variable(:probs), Constant(0.1)])
result = evaluate(tree, ctx)
```

This pattern enables:
- **Aggregator discovery**: Operators working on prediction vectors
- **Belief updating**: Operators accessing running statistics
- **Sequential evaluation**: State that changes between evaluations

For more type-safe code, see `AbstractEvalContext` and `evaluate_with_context`
in [docs/context_aware_evaluation.md](docs/context_aware_evaluation.md).

## Development Status

This package is under active development. Current status:

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Core types & tree utilities | ✅ Complete |
| 2 | Grammar system | ✅ Complete |
| 3 | Evaluation engine | ✅ Complete |
| 4 | Genetic operators | ✅ Complete |
| 5 | NSGA-II optimization | ✅ Complete |
| 6 | DSL interface | ✅ Complete |
| 7 | Pre-built grammar fragments | ⏳ Planned |

## Limitations

**For high-dimensional symbolic regression** (e.g., 100+ variables), consider using
[SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) which
implements multi-population island models optimized for that use case.

SymbolicOptimization.jl excels at:
- Custom objective functions beyond MSE
- Domain-specific operators and grammars
- Context-aware evaluation (belief updating, aggregation)
- Problems with smaller search spaces but complex objectives

## Core Types

### AbstractNode

The base type for all expression tree nodes:

- **`Constant(value::Float64)`** - Literal numeric values
- **`Variable(name::Symbol)`** - Variable references (untyped)
- **`Variable(name::Symbol, type)`** - Variable references (typed)
- **`FunctionNode(func::Symbol, children...)`** - Function applications

### Tree Utilities

```julia
# Creation and copying
copy_tree(node)              # Deep copy

# Size metrics
count_nodes(node)            # Total node count
tree_depth(node)             # Maximum depth
tree_size_stats(node)        # Comprehensive statistics

# Traversal
flatten_tree(node)           # All nodes in pre-order
indexed_nodes(node)          # (index, node) pairs
terminals(node)              # Leaf nodes only
nonterminals(node)           # Internal nodes only

# Collection
collect_variables(node)      # Set of variable names
collect_constants(node)      # Vector of constant values
collect_functions(node)      # Vector of function symbols

# Manipulation
replace_subtree(tree, old, new)  # Substitute subtrees
map_tree(f, node)                # Transform all nodes
get_subtree_at_index(node, i)    # Access by index
random_subtree(node)             # Random selection
```

### Printing

```julia
node_to_string(node)         # Human-readable string
node_to_latex(node)          # LaTeX math notation
print_tree(node)             # Hierarchical ASCII view
```

## Examples

See the `examples/` directory:

**Getting Started:**
- `dsl_example.jl` - **Start here!** User-friendly DSL interface
- `basic_trees.jl` - Working with expression trees
- `grammar_example.jl` - Grammar system usage
- `evaluation_example.jl` - Evaluating expression trees

**Advanced:**
- `genetic_operators_example.jl` - Evolution operators
- `optimization_example.jl` - Multi-objective optimization
- `symbolic_regression_discovery.jl` - Classic symbolic regression examples

**Research Applications:**
- `aggregator_discovery.jl` - Finding crowd wisdom aggregators
- `aggregator_discovery_simple.jl` - Simplified version using DSL
- `belief_updating_discovery.jl` - Context-aware belief updating heuristics
- `confirmation_tracking.jl` - **Policy problem** discovering confirmation measures

## NSGA-II Multi-Objective Optimization

The core API is `optimize(grammar, objectives, data)` — everything else is built on this.

### General Optimization

```julia
# Define what expressions can look like
grammar = Grammar(
    binary_operators = [+, -, *, /],
    unary_operators = [sin, cos],
    variables = [:x, :y],
    constant_range = (-5.0, 5.0),
)

# Define what to optimize (can be ANY computable objectives)
objectives = [
    custom_objective(:my_metric, (tree, data) -> compute_score(tree, data)),
    complexity_objective(),  # Prefer simpler expressions
]

# Provide context data your objectives need
data = Dict(:training_set => ..., :validation_set => ...)

# Run optimization
config = NSGAIIConfig(population_size=100, max_generations=50)
result = optimize(grammar, objectives, data; config=config)

# Analyze Pareto front (trade-off solutions)
for ind in get_pareto_front(result)
    println("Objectives: $(ind.objectives)")
    println("Expression: $(node_to_string(ind.tree))")
end
```

### Curve Fitting (Symbolic Regression)

For the special case of fitting y ≈ f(x), there's a convenience function:

```julia
x = collect(-2.0:0.1:2.0)
y = @. x^2 + 2x + 1  # Target function

result = curve_fitting(x, y;
    config = NSGAIIConfig(population_size=100, max_generations=50)
)

best = get_best(result, 1)  # Best for MSE
println(node_to_string(best.tree))
```

### Built-in Objectives

```julia
mse_objective()           # Mean squared error (for curve fitting)
mae_objective()           # Mean absolute error
complexity_objective()    # Number of nodes
depth_objective()         # Tree depth
custom_objective(name, f) # Your own: f(tree, data) -> Float64
```

## Genetic Operators

Generate, mutate, and crossover expression trees:

```julia
# Generate random trees
tree = generate_tree(grammar; method=GrowMethod(), max_depth=4)
pop = generate_population(grammar, 100; method=HalfAndHalfMethod())

# Mutation operators
mutated = mutate(tree, grammar)                    # Auto-select mutation type
mutated = mutate_point(tree, grammar)              # Swap operator/variable
mutated = mutate_subtree(tree, grammar)            # Replace subtree
mutated = mutate_hoist(tree)                       # Promote subtree to root
mutated = mutate_constants(tree; std=0.5)          # Perturb constants
mutated = mutate_insert(tree, grammar)             # Add operator
mutated = mutate_delete(tree)                      # Remove operator

# Crossover operators
child1, child2 = crossover(parent1, parent2)       # Subtree exchange
child1, child2 = crossover_size_fair(p1, p2)       # Similar-size subtrees

# Simplification
simplified = simplify(tree; grammar=grammar)        # Constant folding + algebra
simplified = simplify_algebra(tree)                 # x + 0 → x, x * 1 → x, etc.
simplified = simplify_constants(tree)               # 1 + 2 → 3

# Deep simplification (requires `using Symbolics`)
simplified = deep_simplify(tree)                    # Full CAS simplification
str = simplified_string(tree)                       # Simplified as string
tex = simplified_latex(tree)                        # Simplified as LaTeX
r = simplify_piecewise(tree)                        # Piecewise formula simplification
r = simplify_piecewise(tree, substitutions =        # With domain constraints
        complement_vars(:pnotH => :pH))
```

## Running Tests

```julia
using Pkg
Pkg.test("SymbolicOptimization")
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
