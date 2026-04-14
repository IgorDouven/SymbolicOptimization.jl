# Grammar System

The grammar system defines the search space for symbolic optimization: which operators, variables, and constants are available, and how they can be combined.

## Simple Grammar (Untyped)

For standard symbolic regression:

```julia
grammar = Grammar(
    binary_operators = [+, -, *, /],
    unary_operators = [sin, cos, exp, log],
    variables = [:x, :y],
    constant_range = (-2.0, 2.0),
)
```

## Typed Grammar

For domains with multiple types (e.g., vector/scalar operations):

```julia
grammar = Grammar(
    types = [:Scalar, :Vector],
    variables = [
        :ps => :Vector,
        :n => :Scalar,
    ],
    operators = [
        (mean, [:Vector] => :Scalar),
        (sum, [:Vector] => :Scalar),
        (+, [:Scalar, :Scalar] => :Scalar),
        (*, [:Scalar, :Scalar] => :Scalar),
        (Symbol(".^"), [:Vector, :Scalar] => :Vector),
    ],
    constant_types = [:Scalar],
    output_type = :Scalar,
)
```

## Querying the Grammar

```julia
unary_operators(grammar)                # Unary operators
binary_operators(grammar)               # Binary operators
operators_producing(grammar, :Scalar)   # Operators producing a type
sample_constant(grammar)                # Random constant

valid, msg = check_tree_validity(tree, grammar)  # Validate a tree
```

## Safe Operations

All standard operators have "safe" versions that return `NaN` instead of throwing:

```julia
safe_div(1.0, 0.0)    # NaN
safe_log(-1.0)        # NaN
safe_sqrt(-1.0)       # NaN
safe_pow(-2.0, 0.5)   # NaN
```

Additional safe functions include `safe_exp`, `safe_mean`, `safe_sum`, `safe_std`, `safe_var`, and activation functions like `sigmoid`, `relu`, `softplus`, `clamp01`.

The `SAFE_IMPLEMENTATIONS` dictionary maps standard function symbols to their safe counterparts.

## Grammar Reference

```@docs
Grammar
OperatorSpec
VariableSpec
ConstantSpec
ValidationResult
is_typed
all_operators
all_variables
all_constants
operators_by_name
operators_by_arity
operators_producing
unary_operators
binary_operators
ternary_operators
variables_of_type
constants_of_type
has_type
num_operators
num_variables
sample_constant
validate_grammar
validate_grammar!
check_tree_validity
infer_type
```
