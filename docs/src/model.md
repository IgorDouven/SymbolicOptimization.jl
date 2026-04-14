# Model API

The Model API provides a JuMP-inspired interface using [`SymbolicModel`](@ref) and macros for defining symbolic optimization problems.

## Basic Usage

```julia
using SymbolicOptimization

model = SymbolicModel()

@variables model x y
@operators model [+, -, *, /] [sin, cos]
@data model X y_data
@objective model mse
@objective model complexity
@config model population=200 generations=100

optimize!(model)

println(expression_string(model))
println(objective_value(model))
```

## Macros

### `@variables`

Define the variable names for the search space:

```julia
@variables model x y z
```

### `@operators`

Set binary and unary operators:

```julia
@operators model [+, -, *, /] [sin, cos, exp]
```

### `@constants`

Configure the constant range:

```julia
@constants model (-5.0, 5.0)
```

### `@objective`

Add optimization objectives:

```julia
@objective model mse
@objective model complexity
```

### `@constraint`

Add constraints to the search:

```julia
@constraint model max_depth 6
@constraint model max_nodes 30
```

### `@seed`

Provide seed expressions:

```julia
@seed model expr_tree
```

### `@data`

Provide training data:

```julia
@data model X_matrix y_vector
```

### `@config`

Set configuration options:

```julia
@config model population=200 generations=100 seed=42
```

### `@grammar`

Define a full grammar directly:

```julia
@grammar model my_grammar
```

### `@problem`

Composite macro that combines multiple definitions:

```julia
@problem model begin
    @variables x y
    @operators [+, -, *] [sin]
    @data X y
end
```

## Programmatic API

For cases where macros are less convenient:

```julia
model = SymbolicModel()
set_variables!(model, [:x, :y])
set_operators!(model, binary=[+, -, *], unary=[sin])
set_constants!(model, (-5.0, 5.0))
set_data!(model, X, y)
set_config!(model, population=200)
add_constraint!(model, :max_depth, 6)
add_seed!(model, seed_tree)

optimize!(model)
```

## Inspecting Results

```julia
expression_string(model)   # Best expression as string
expression_latex(model)    # Best expression as LaTeX
predict(model, X_new)      # Predictions on new data
objective_value(model)     # Objective values
history(model)             # Optimization history
raw_result(model)          # Full NSGAIIResult
```

## Model API Reference

```@docs
SymbolicModel
optimize!
set_variables!
set_operators!
set_constants!
set_data!
set_config!
add_constraint!
set_constraint_mode!
add_seed!
expression_string
expression_latex
predict
objective_value
history
raw_result
expr_to_tree
```
