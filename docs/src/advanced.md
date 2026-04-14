# Advanced Topics

## Deep Algebraic Simplification (Symbolics.jl)

The built-in [`simplify`](@ref) handles basic identities (`x + 0 -> x`, `x * 1 -> x`) and is applied during evolution. For deeper algebraic simplification of final results, load `Symbolics.jl`:

```julia
using SymbolicOptimization
using Symbolics  # activates the extension

b = best(result)

# Deeply simplified tree (still an AbstractNode)
simple_tree = deep_simplify(b.tree)

# Or get simplified output directly
simplified_string(b.tree)   # "x^2 - 1"
simplified_latex(b.tree)    # "x^{2} - 1"
```

Safe operators (`safe_div`, `safe_log`, etc.) are automatically mapped to their standard equivalents before simplification.

### Piecewise Formulas

If your grammar includes `step_func`, the search may discover piecewise formulas. Use [`simplify_piecewise`](@ref) to simplify each branch independently:

```julia
r = simplify_piecewise(b.tree)

r.if_string      # simplified if-branch as string
r.else_string    # simplified else-branch as string
r.if_latex       # LaTeX for the if-branch
r.else_latex     # LaTeX for the else-branch
r.if_branch      # simplified if-branch as AbstractNode
r.else_branch    # simplified else-branch as AbstractNode
```

### Domain Substitutions

Declare domain-specific identities via the `substitutions` argument:

```julia
r = simplify_piecewise(b.tree, substitutions = complement_vars(
    :pnotH_notE => :pH_notE,   # P(not H|not E) = 1 - P(H|not E)
    :pnotH_E    => :pH_E,      # P(not H|E) = 1 - P(H|E)
))
```

### Limitations

- **Factoring**: Symbolics.jl tends to expand rather than factor polynomials.
- **Piecewise detection**: Assumes all `step_func(...)` calls share the same condition.
- **Performance**: Deep simplification should only be used on final results, not inside the GP loop.

If you don't load `Symbolics`, none of these functions are available, keeping the package lightweight.

## Constraints

Define constraints to guide the search toward valid expressions:

```julia
cs = ConstraintSet()
add_constraint!(cs, directionality_constraint(:x, :increasing))
add_constraint!(cs, monotonicity_constraint(:y, data))
add_constraint!(cs, boundary_constraint(:x, 0.0, 1.0))

violation = check_constraints(tree, cs, data)
rate = violation_rate(tree, cs, data)
```

### Built-in Constraints

- [`directionality_constraint`](@ref) — enforce monotonic relationships
- [`logicality_constraint`](@ref) — logical consistency
- [`symmetry_constraint`](@ref) — symmetry properties
- [`monotonicity_constraint`](@ref) — monotonicity on data
- [`boundary_constraint`](@ref) — output bounds
- [`confirmation_measure_constraints`](@ref) — domain-specific constraints for confirmation measures

## Policy Problems

For problems where objectives cannot be computed point-by-point, use [`policy_problem`](@ref) with a custom evaluator.

### Discrimination Problems (AUC-based)

```julia
function my_simulator(rng)
    inputs = Dict(:x => rand(rng), :y => rand(rng))
    label = some_ground_truth
    return (inputs, label)
end

result = solve(policy_problem(
    variables = [:x, :y],
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

```julia
my_evaluator = (tree, env, evaluate_fn, count_nodes_fn) -> begin
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

## Advanced Reference

```@docs
deep_simplify
simplified_string
simplified_latex
simplify_piecewise
PiecewiseResult
complement_vars
to_symbolics
from_symbolics
Constraint
ConstraintSet
check_constraint
check_constraints
violation_rate
directionality_constraint
logicality_constraint
symmetry_constraint
monotonicity_constraint
boundary_constraint
confirmation_measure_constraints
policy_problem
discrimination_evaluator
sequential_evaluator
calibration_evaluator
```
