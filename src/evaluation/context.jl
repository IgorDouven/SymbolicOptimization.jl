# ═══════════════════════════════════════════════════════════════════════════════
# Domain-Specific Evaluation Contexts
# ═══════════════════════════════════════════════════════════════════════════════
#
# This module provides an extensible evaluation system that supports:
# 1. Stateless batch evaluation (e.g., aggregator discovery)
# 2. Stateful sequential evaluation (e.g., belief updating)
# 3. Custom operators that depend on external context
#
# Users define domain-specific contexts by:
# 1. Creating a subtype of AbstractEvalContext
# 2. Implementing resolve_variable(ctx, name)
# 3. Optionally implementing apply_operator(ctx, op, args...) for custom operators

"""
    AbstractEvalContext

Abstract base type for evaluation contexts. Subtype this to create
domain-specific evaluation contexts.

# Required Interface

Subtypes should implement:
- `resolve_variable(ctx::MyContext, name::Symbol) -> Any`

# Optional Interface

Subtypes may implement:
- `apply_operator(ctx::MyContext, op::Symbol, args...) -> Any`
- `has_custom_operator(ctx::MyContext, op::Symbol) -> Bool`

# Example: Aggregator Context

```julia
struct AggregatorContext <: AbstractEvalContext
    ps::Vector{Float64}  # Forecaster predictions
end

resolve_variable(ctx::AggregatorContext, name::Symbol) = 
    name == :ps ? ctx.ps : error("Unknown variable: \$name")
```

# Example: Belief Updating Context

```julia
mutable struct BeliefContext <: AbstractEvalContext
    probs::Vector{Float64}
    data_mean::Float64
    is_heads::Bool
    # ... more fields
end

resolve_variable(ctx::BeliefContext, name::Symbol) = begin
    name == :probs && return ctx.probs
    name == :data_mean && return ctx.data_mean
    # ...
end

# Custom operator that uses context
apply_operator(ctx::BeliefContext, op::Symbol, args...) = begin
    if op == :add_ibe_bonus
        return add_ibe_bonus_impl(args[1], args[2], ctx.data_mean)
    elseif op == :select_by_data
        return ctx.is_heads ? args[1] : args[2]
    else
        return nothing  # Fall back to default
    end
end

has_custom_operator(ctx::BeliefContext, op::Symbol) = 
    op in (:add_ibe_bonus, :select_by_data)
```
"""
abstract type AbstractEvalContext end

# ───────────────────────────────────────────────────────────────────────────────
# Interface Functions
# ───────────────────────────────────────────────────────────────────────────────

"""
    resolve_variable(ctx::AbstractEvalContext, name::Symbol) -> Any

Resolve a variable name to its value in the given context.
Must be implemented by subtypes.
"""
function resolve_variable(ctx::AbstractEvalContext, name::Symbol)
    error("resolve_variable not implemented for $(typeof(ctx))")
end

"""
    apply_operator(ctx::AbstractEvalContext, op::Symbol, args...) -> Any

Apply an operator with the given arguments in the context.
Return `nothing` to use the default implementation.

Default implementation returns `nothing`, causing fallback to standard evaluation.
"""
function apply_operator(ctx::AbstractEvalContext, op::Symbol, args...)
    return nothing  # Signal to use default
end

"""
    has_custom_operator(ctx::AbstractEvalContext, op::Symbol) -> Bool

Check if the context provides a custom implementation for the operator.
Default returns `false`.
"""
function has_custom_operator(ctx::AbstractEvalContext, op::Symbol)
    return false
end

"""
    has_variable(ctx::AbstractEvalContext, name::Symbol) -> Bool

Check if a variable exists in the context.
Default implementation tries to resolve and catches errors.
"""
function has_variable(ctx::AbstractEvalContext, name::Symbol)
    try
        resolve_variable(ctx, name)
        return true
    catch
        return false
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Make EvalContext a subtype (for backward compatibility)
# ───────────────────────────────────────────────────────────────────────────────

# Note: We can't make EvalContext <: AbstractEvalContext without modifying it,
# so we provide adapter methods instead.

"""
    resolve_variable(ctx::EvalContext, name::Symbol)

Resolve variable for the standard EvalContext.
"""
resolve_variable(ctx::EvalContext, name::Symbol) = ctx[name]

has_variable(ctx::EvalContext, name::Symbol) = haskey(ctx, name)

# ───────────────────────────────────────────────────────────────────────────────
# Context-Aware Evaluation
# ───────────────────────────────────────────────────────────────────────────────

"""
    evaluate_with_context(tree::AbstractNode, ctx) -> Any

Evaluate a tree using a custom evaluation context.

This is the main entry point for domain-specific evaluation.
The context controls:
- How variables are resolved
- How operators are applied (with optional custom implementations)

# Example

```julia
# Define your context
struct MyContext <: AbstractEvalContext
    x::Float64
    y::Float64
end

resolve_variable(ctx::MyContext, name::Symbol) = 
    name == :x ? ctx.x : name == :y ? ctx.y : error("Unknown: \$name")

# Evaluate
ctx = MyContext(1.0, 2.0)
tree = FunctionNode(:+, [Variable(:x), Variable(:y)])
result = evaluate_with_context(tree, ctx)  # 3.0
```
"""
function evaluate_with_context(tree::AbstractNode, ctx)
    return _eval_with_ctx(tree, ctx)
end

function _eval_with_ctx(node::Constant, ctx)
    return node.value
end

function _eval_with_ctx(node::Variable, ctx)
    return resolve_variable(ctx, node.name)
end

function _eval_with_ctx(node::FunctionNode, ctx)
    # Evaluate children first
    args = [_eval_with_ctx(child, ctx) for child in node.children]
    
    # Try custom operator first
    if has_custom_operator(ctx, node.func)
        result = apply_operator(ctx, node.func, args...)
        if result !== nothing
            return result
        end
    end
    
    # Also try apply_operator even without has_custom_operator
    # (for simpler implementations that just return nothing for unknown ops)
    result = apply_operator(ctx, node.func, args...)
    if result !== nothing
        return result
    end
    
    # Fall back to default operator implementation
    return _apply_default_operator(node.func, args)
end

"""
    _apply_default_operator(op::Symbol, args) -> Any

Apply a standard operator. Used as fallback when context doesn't provide custom implementation.
"""
function _apply_default_operator(op::Symbol, args)
    # Check safe implementations
    if haskey(SAFE_IMPLEMENTATIONS, op)
        return SAFE_IMPLEMENTATIONS[op](args...)
    end
    
    # Standard operators
    op == :+ && return +(args...)
    op == :- && return length(args) == 1 ? -args[1] : -(args...)
    op == :* && return *(args...)
    op == :/ && return safe_div(args...)
    op == :^ && return safe_pow(args...)
    
    # Math functions
    op == :sin && return sin(args[1])
    op == :cos && return cos(args[1])
    op == :tan && return safe_tan(args[1])
    op == :exp && return safe_exp(args[1])
    op == :log && return safe_log(args[1])
    op == :sqrt && return safe_sqrt(args[1])
    op == :abs && return abs(args[1])
    
    # Aggregations
    op == :mean && return safe_mean(args[1])
    op == :sum && return safe_sum(args[1])
    op == :std && return safe_std(args[1])
    op == :var && return safe_var(args[1])
    op == :maximum && return maximum(args[1])
    op == :minimum && return minimum(args[1])
    
    # Activations
    op == :sigmoid && return sigmoid(args[1])
    op == :relu && return relu(args[1])
    op == :tanh && return tanh(args[1])
    op == :clamp01 && return clamp01(args[1])
    
    # Vector operations (element-wise)
    op == :ew_add && return args[1] .+ args[2]
    op == :ew_sub && return args[1] .- args[2]
    op == :ew_mul && return args[1] .* args[2]
    op == :ew_div && return args[1] ./ (args[2] .+ 1e-10)
    op == :ew_pow && return (abs.(args[1]) .+ 1e-10) .^ args[2]
    
    # Scalar-vector operations
    op == :sv_mul && return args[1] .* args[2]
    op == :sv_add && return args[1] .+ args[2]
    op == :sv_pow && return (abs.(args[2]) .+ 1e-10) .^ args[1]
    
    # Vector functions
    op == :normalize && return begin
        v = args[1]
        v = max.(v, 0.0)
        s = sum(v)
        s > 0 ? v ./ s : v
    end
    
    op == :dot && return LinearAlgebra.dot(args[1], args[2])
    
    # Weighted sum: w*a + (1-w)*b
    op == :weighted_sum && return begin
        w = clamp(args[1], 0.0, 1.0)
        w .* args[2] .+ (1.0 - w) .* args[3]
    end
    
    # Try Base
    if isdefined(Base, op)
        f = getfield(Base, op)
        return f(args...)
    end
    
    error("Unknown operator: $op (args: $(length(args)))")
end

# ───────────────────────────────────────────────────────────────────────────────
# Safe Evaluation with Context
# ───────────────────────────────────────────────────────────────────────────────

"""
    safe_evaluate_with_context(tree::AbstractNode, ctx; default=NaN) -> Any

Safely evaluate a tree, returning `default` on any error.
"""
function safe_evaluate_with_context(tree::AbstractNode, ctx; default=NaN)
    try
        result = evaluate_with_context(tree, ctx)
        # Check for NaN/Inf in numeric results
        if result isa Number && !isfinite(result)
            return default
        end
        return result
    catch
        return default
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Sequential Evaluation Helper
# ───────────────────────────────────────────────────────────────────────────────

"""
    evaluate_sequential(tree::AbstractNode, make_context, update_context!, n_steps;
                        init_state=nothing) -> Vector

Evaluate a tree sequentially over multiple steps.

# Arguments
- `tree`: The expression tree to evaluate
- `make_context`: Function `(state, step) -> context` that creates evaluation context
- `update_context!`: Function `(state, result, step) -> nothing` that updates state after evaluation
- `n_steps`: Number of steps to run
- `init_state`: Initial state passed to make_context

# Returns
Vector of results from each step.

# Example: Belief Updating

```julia
# State tracks current beliefs and data
mutable struct State
    probs::Vector{Float64}
    data::Vector{Bool}
end

make_ctx = (state, t) -> BeliefContext(state.probs, state.data, t, ...)
update_state! = (state, result, t) -> begin
    state.probs = normalize(result)
end

results = evaluate_sequential(tree, make_ctx, update_state!, 250, init_state=State(...))
```
"""
function evaluate_sequential(tree::AbstractNode, make_context, update_state!, n_steps;
                             init_state=nothing)
    results = []
    state = init_state
    
    for step in 1:n_steps
        ctx = make_context(state, step)
        result = safe_evaluate_with_context(tree, ctx)
        push!(results, result)
        update_state!(state, result, step)
    end
    
    return results
end

# ───────────────────────────────────────────────────────────────────────────────
# Pre-built Context Types
# ───────────────────────────────────────────────────────────────────────────────

"""
    SimpleContext(; kwargs...)
    SimpleContext(bindings::Dict)

A simple context that just holds variable bindings.
This is the context-system equivalent of EvalContext.

# Example

```julia
ctx = SimpleContext(x=1.0, y=2.0)
result = evaluate_with_context(tree, ctx)
```
"""
struct SimpleContext <: AbstractEvalContext
    bindings::Dict{Symbol, Any}
end

SimpleContext(; kwargs...) = SimpleContext(Dict{Symbol,Any}(kwargs...))
SimpleContext(d::Dict) = SimpleContext(Dict{Symbol,Any}(d))

resolve_variable(ctx::SimpleContext, name::Symbol) = 
    haskey(ctx.bindings, name) ? ctx.bindings[name] : error("Unknown variable: $name")

has_variable(ctx::SimpleContext, name::Symbol) = haskey(ctx.bindings, name)

"""
    VectorAggregatorContext

Pre-built context for aggregator discovery problems.
Provides a prediction vector `ps` and common vector operations.

# Example

```julia
ctx = VectorAggregatorContext(predictions)
result = evaluate_with_context(aggregator_tree, ctx)
```
"""
struct VectorAggregatorContext <: AbstractEvalContext
    ps::Vector{Float64}
    extra::Dict{Symbol, Any}
end

VectorAggregatorContext(ps::Vector{Float64}) = VectorAggregatorContext(ps, Dict{Symbol,Any}())

function resolve_variable(ctx::VectorAggregatorContext, name::Symbol)
    name == :ps && return ctx.ps
    haskey(ctx.extra, name) && return ctx.extra[name]
    error("Unknown variable: $name")
end

has_variable(ctx::VectorAggregatorContext, name::Symbol) = 
    name == :ps || haskey(ctx.extra, name)
