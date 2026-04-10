# API Reference

## Core Types

```@docs
AbstractNode
Constant
Variable
FunctionNode
```

## Tree Utilities

```@docs
copy_tree
count_nodes
tree_depth
flatten_tree
collect_variables
collect_constants
collect_functions
replace_subtree
map_tree
get_subtree_at_index
indexed_nodes
random_subtree
terminals
nonterminals
tree_size_stats
```

## Printing

```@docs
node_to_string
node_to_latex
print_tree
```

## Grammar

```@docs
Grammar
OperatorSpec
VariableSpec
ConstantSpec
validate_grammar
register_safe_op!
```

## Evaluation

```@docs
EvalContext
evaluate
evaluate_batch
safe_evaluate
compile_tree
AbstractEvalContext
SimpleContext
VectorAggregatorContext
evaluate_with_context
evaluate_sequential
```

## Genetic Operators

```@docs
GenerationMethod
FullMethod
GrowMethod
HalfAndHalfMethod
generate_tree
generate_population
mutate
crossover
simplify
```

## NSGA-II Optimization

```@docs
Individual
ObjectiveFunction
NSGAIIConfig
NSGAIIResult
mse_objective
mae_objective
complexity_objective
depth_objective
custom_objective
optimize
get_best
get_pareto_front
curve_fitting
```

## DSL Interface

```@docs
SymbolicProblem
SymbolicResult
PolicyProblem
symbolic_problem
policy_problem
variables!
operators!
constants!
add_function!
objectives!
add_objective!
data!
config!
mode!
solve
best
pareto_front
evaluate_best
```

## JuMP-style API

```@docs
SymbolicModel
optimize!
expression_string
expression_latex
predict
objective_value
expr_to_tree
```

## Evaluators

```@docs
discrimination_evaluator
sequential_evaluator
calibration_evaluator
compute_auc
compute_brier
compute_log_score
```

## Constraints

```@docs
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
```

## Symbolics.jl Integration

These functions require `using Symbolics` to activate the package extension.

```@docs
to_symbolics
from_symbolics
deep_simplify
simplified_string
simplified_latex
simplify_piecewise
PiecewiseResult
complement_vars
```
