# Context-Aware Evaluation

SymbolicOptimization provides flexible evaluation systems for expression trees that need access to external context. This is essential for problems like:

- **Aggregator discovery**: Operators that work on prediction vectors
- **Belief updating**: Operators that depend on running statistics (data mean, observation history)
- **Time-series processing**: Operators that access historical values
- **Reinforcement learning**: Operators that read environment state

## Two Approaches

### Approach 1: EvalContext with Operators Dict (Simple)

For quick prototyping, use `EvalContext` with a custom operators dictionary:

```julia
using SymbolicOptimization

# Variable bindings
bindings = Dict{Symbol, Any}(
    :probs => [0.1, 0.2, 0.7],
    :data_mean => 0.6,
    :is_heads => true
)

# Custom operators - receive (args..., ctx) 
operators = Dict{Symbol, Function}(
    # Operator that uses context
    :add_bonus => (v, bonus, ctx) -> begin
        dm = ctx[:data_mean]  # Access context variables
        result = copy(v)
        idx = round(Int, dm * (length(v) - 1)) + 1
        result[idx] += bonus
        return result
    end,
    
    # Operator with conditional logic from context
    :choose_by_obs => (a, b, ctx) -> ctx[:is_heads] ? a : b,
    
    # Pure operator (ignores context)
    :normalize => (v, ctx) -> v ./ sum(v)
)

# Create context
ctx = EvalContext(bindings, operators)

# Build tree
tree = FunctionNode(:normalize, [
    FunctionNode(:add_bonus, [Variable(:probs), Constant(0.1)])
])

# Evaluate
result = evaluate(tree, ctx)
```

**Helper functions:**
- `with_operators(ctx, ops)` - Add operators to existing context
- `with_bindings(ctx, bindings)` - Add bindings to existing context
- `has_operator(ctx, :name)` - Check if operator exists
- `get_operator(ctx, :name)` - Get operator function

### Approach 2: AbstractEvalContext (Type-Safe)

For production code with complex domains, define a typed context:

```julia
using SymbolicOptimization

# Define your context type
struct BeliefContext <: AbstractEvalContext
    probs::Vector{Float64}
    data_mean::Float64
    n_tosses::Int
    is_heads::Bool
end

# Implement variable resolution (required)
function SymbolicOptimization.resolve_variable(ctx::BeliefContext, name::Symbol)
    name == :probs && return ctx.probs
    name == :data_mean && return ctx.data_mean
    name == :n_tosses && return Float64(ctx.n_tosses)
    error("Unknown variable: $name")
end

# Implement custom operators (optional)
function SymbolicOptimization.apply_operator(ctx::BeliefContext, op::Symbol, args...)
    if op == :add_bonus
        v, bonus = args
        idx = round(Int, ctx.data_mean * (length(v) - 1)) + 1
        result = copy(v)
        result[idx] += bonus
        return result
    elseif op == :choose_by_obs
        return ctx.is_heads ? args[1] : args[2]
    end
    return nothing  # Fall back to default operators
end

# Mark which operators are custom
function SymbolicOptimization.has_custom_operator(ctx::BeliefContext, op::Symbol)
    op in (:add_bonus, :choose_by_obs)
end

# Use with evaluate_with_context
ctx = BeliefContext([0.1, 0.2, 0.7], 0.6, 100, true)
result = evaluate_with_context(tree, ctx)
```

## Sequential Evaluation

For iterative problems (like belief updating), use `evaluate_sequential`:

```julia
# State that persists across steps
mutable struct SimState
    probs::Vector{Float64}
    observations::Vector{Bool}
end

# Create context for each step
make_ctx = (state, step) -> EvalContext(
    Dict(:probs => state.probs, :obs => state.observations[step]),
    operators
)

# Update state after each step
update! = (state, result, step) -> begin
    state.probs = normalize(result)
end

# Run 100 steps
state = SimState(prior, observations)
results = evaluate_sequential(tree, make_ctx, update!, 100; init_state=state)
```

## Comparison

| Feature | EvalContext + Dict | AbstractEvalContext |
|---------|-------------------|---------------------|
| Setup complexity | Low | Medium |
| Type safety | Low | High |
| Performance | Good | Better (specialized methods) |
| Extensibility | Limited | Excellent |
| Multiple dispatch | No | Yes |
| Best for | Prototyping, simple cases | Production, complex domains |

## Examples

See the examples directory:
- `aggregator_discovery.jl` - Uses simple EvalContext
- `belief_updating_discovery.jl` - Uses EvalContext with custom operators

## Built-in Default Operators

When no custom operator is found, the system falls back to built-in operators:

**Arithmetic:** `+`, `-`, `*`, `/`, `^`, `neg`, `inv`  
**Math:** `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`  
**Aggregation:** `mean`, `sum`, `std`, `var`, `maximum`, `minimum`  
**Activation:** `sigmoid`, `relu`, `tanh`, `clamp01`  
**Vector:** `ew_add`, `ew_sub`, `ew_mul`, `ew_div`, `ew_pow`, `sv_mul`, `sv_add`, `normalize`, `dot`, `weighted_sum`

All built-in operators handle edge cases safely (division by zero returns safe values, etc.).
