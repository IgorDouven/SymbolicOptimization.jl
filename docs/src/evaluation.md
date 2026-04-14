# Evaluation

The evaluation engine computes the value of expression trees given variable bindings.

## Basic Evaluation

```julia
tree = FunctionNode(:+, Variable(:x), Constant(1.0))

# Multiple calling conventions
evaluate(tree, x=5.0)                           # 6.0
evaluate(tree, EvalContext(x=5.0))              # 6.0
evaluate(tree, Dict(:x => 5.0))                 # 6.0
evaluate(tree, grammar, EvalContext(x=5.0))     # Uses grammar's safe operators
```

## Batch Evaluation

```julia
data = [1.0; 2.0; 3.0;;]  # 3x1 matrix
results = evaluate_batch(tree, data, [:x])      # [2.0, 3.0, 4.0]
```

## Safe Evaluation

```julia
safe_evaluate(tree, EvalContext(x=5.0))   # Returns NaN on any error
is_valid_on(tree, EvalContext(x=5.0))     # true if evaluation succeeds
```

## Compiled Evaluation

For evaluating the same tree many times, compile it to a function:

```julia
f = compile_tree(tree, [:x])
f(5.0)  # 6.0
```

## Context-Aware Evaluation

For problems where operators need access to external state (belief updating, aggregation, time-series), use context-aware evaluation.

### EvalContext with Custom Operators

```julia
operators = Dict{Symbol, Function}(
    :add_bonus => (v, bonus, ctx) -> begin
        dm = ctx[:data_mean]
        result = copy(v)
        idx = argmin(abs.(v .- dm))
        result[idx] += bonus
        return result
    end,
    :choose_by_obs => (a, b, ctx) -> ctx[:is_heads] ? a : b,
)

ctx = EvalContext(
    Dict{Symbol, Any}(:probs => probs, :data_mean => 0.6, :is_heads => true),
    operators
)

result = evaluate(tree, ctx)
```

Helper functions:
- [`with_operators`](@ref) — add operators to existing context
- [`with_bindings`](@ref) — add bindings to existing context
- [`has_operator`](@ref) — check if operator exists
- [`get_operator`](@ref) — get operator function

### AbstractEvalContext (Type-Safe)

For production code, define a typed context:

```julia
struct BeliefContext <: AbstractEvalContext
    probs::Vector{Float64}
    data_mean::Float64
    n_tosses::Int
    is_heads::Bool
end

function SymbolicOptimization.resolve_variable(ctx::BeliefContext, name::Symbol)
    name == :probs && return ctx.probs
    name == :data_mean && return ctx.data_mean
    error("Unknown variable: \$name")
end

function SymbolicOptimization.apply_operator(ctx::BeliefContext, op::Symbol, args...)
    if op == :add_bonus
        v, bonus = args
        result = copy(v)
        result[argmin(abs.(v .- ctx.data_mean))] += bonus
        return result
    end
    return nothing  # Fall back to defaults
end

function SymbolicOptimization.has_custom_operator(ctx::BeliefContext, op::Symbol)
    op in (:add_bonus,)
end

ctx = BeliefContext([0.1, 0.2, 0.7], 0.6, 100, true)
result = evaluate_with_context(tree, ctx)
```

### Sequential Evaluation

For iterative problems (belief updating, time-series):

```julia
results = evaluate_sequential(tree, make_ctx, update!, n_steps; init_state=state)
```

## Evaluation Reference

```@docs
evaluate
evaluate_batch
safe_evaluate
is_valid_on
compile_tree
EvalContext
with_operators
with_bindings
has_operator
get_operator
AbstractEvalContext
SimpleContext
VectorAggregatorContext
resolve_variable
apply_operator
has_custom_operator
has_variable
evaluate_with_context
safe_evaluate_with_context
evaluate_sequential
```
