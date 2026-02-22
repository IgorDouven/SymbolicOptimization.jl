# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Engine
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
# Evaluation Context
# ───────────────────────────────────────────────────────────────────────────────

"""
    EvalContext

A context for evaluating expression trees, binding variable names to values
and optionally providing custom operator implementations.

# Basic Usage (Variable Bindings)

```julia
# From keyword arguments
ctx = EvalContext(x=1.0, y=2.0)

# From a Dict
ctx = EvalContext(Dict(:x => 1.0, :y => 2.0))

# From pairs
ctx = EvalContext(:x => 1.0, :y => 2.0)

ctx[:x]  # 1.0
ctx[:z]  # throws KeyError
haskey(ctx, :x)  # true
```

# Advanced Usage (Context-Aware Operators)

For problems like belief updating where operators need access to context
(e.g., `add_ibe_bonus` needs `data_mean`), you can register custom operators
that receive the full context:

```julia
# Define context-aware operators
# Signature: (args..., ctx::EvalContext) -> result
custom_ops = Dict{Symbol, Function}(
    :add_ibe_bonus => (v, bonus, ctx) -> begin
        data_mean = ctx[:data_mean]
        # Find hypothesis closest to data_mean, add bonus to it
        biases = ctx[:biases]
        idx = argmin(abs.(biases .- data_mean))
        result = copy(v)
        result[idx] += bonus
        return result
    end,
    :select_by_data => (a, b, ctx) -> ctx[:is_heads] ? a : b
)

# Create context with operators
ctx = EvalContext(
    Dict(:probs => probs, :data_mean => 0.6, :is_heads => true, :biases => 0:0.1:1),
    custom_ops
)

# Evaluate - custom operators automatically receive context
result = evaluate(tree, ctx)
```

This allows the same tree structure to be used for:
- **Aggregator discovery**: Simple context with just `ps` (predictions vector)
- **Belief updating**: Rich context with running statistics, custom operators
"""
struct EvalContext
    bindings::Dict{Symbol, Any}
    operators::Dict{Symbol, Function}  # Custom operator implementations (args..., ctx) -> result
end

# Default constructor with empty operators
EvalContext(bindings::Dict{Symbol, Any}) = EvalContext(bindings, Dict{Symbol, Function}())

# Convert any AbstractDict with Symbol keys
function EvalContext(d::AbstractDict{Symbol})
    new_dict = Dict{Symbol, Any}(k => v for (k, v) in d)
    return EvalContext(new_dict, Dict{Symbol, Function}())
end

# Constructor with bindings dict and operators dict
function EvalContext(bindings::AbstractDict{Symbol}, operators::AbstractDict{Symbol, Function})
    new_bindings = Dict{Symbol, Any}(k => v for (k, v) in bindings)
    new_operators = Dict{Symbol, Function}(k => v for (k, v) in operators)
    return EvalContext(new_bindings, new_operators)
end

# At least one pair required to avoid ambiguity with empty kwargs
function EvalContext(p1::Pair{Symbol}, pairs::Pair{Symbol}...)
    d = Dict{Symbol, Any}(p1, pairs...)
    return EvalContext(d, Dict{Symbol, Function}())
end

function EvalContext(; kwargs...)
    d = Dict{Symbol, Any}()
    for (k, v) in kwargs
        d[k] = v
    end
    return EvalContext(d, Dict{Symbol, Function}())
end

Base.getindex(ctx::EvalContext, key::Symbol) = ctx.bindings[key]
Base.setindex!(ctx::EvalContext, value, key::Symbol) = ctx.bindings[key] = value
Base.haskey(ctx::EvalContext, key::Symbol) = haskey(ctx.bindings, key)
Base.keys(ctx::EvalContext) = keys(ctx.bindings)
Base.values(ctx::EvalContext) = values(ctx.bindings)
Base.length(ctx::EvalContext) = length(ctx.bindings)

"""
    has_operator(ctx::EvalContext, op::Symbol) -> Bool

Check if the context has a custom implementation for the given operator.
"""
has_operator(ctx::EvalContext, op::Symbol) = haskey(ctx.operators, op)

"""
    get_operator(ctx::EvalContext, op::Symbol) -> Union{Function, Nothing}

Get the custom operator implementation, or nothing if not defined.
"""
get_operator(ctx::EvalContext, op::Symbol) = get(ctx.operators, op, nothing)

function Base.show(io::IO, ctx::EvalContext)
    print(io, "EvalContext(")
    pairs = ["$k=$(repr(v))" for (k, v) in ctx.bindings]
    print(io, join(pairs, ", "))
    if !isempty(ctx.operators)
        print(io, "; operators=[", join(keys(ctx.operators), ", "), "]")
    end
    print(io, ")")
end

"""
    merge(ctx::EvalContext, other::EvalContext) -> EvalContext
    merge(ctx::EvalContext; kwargs...) -> EvalContext

Create a new context with merged bindings (and operators from first context).
"""
function Base.merge(ctx::EvalContext, other::EvalContext)
    EvalContext(
        merge(ctx.bindings, other.bindings),
        merge(ctx.operators, other.operators)
    )
end

function Base.merge(ctx::EvalContext; kwargs...)
    d = copy(ctx.bindings)
    for (k, v) in kwargs
        d[k] = v
    end
    EvalContext(d, ctx.operators)
end

"""
    with_operators(ctx::EvalContext, ops::Dict{Symbol, Function}) -> EvalContext

Create a new context with additional/overridden operators.
"""
function with_operators(ctx::EvalContext, ops::Dict{Symbol, Function})
    EvalContext(ctx.bindings, merge(ctx.operators, ops))
end

"""
    with_bindings(ctx::EvalContext, bindings::Dict) -> EvalContext
    with_bindings(ctx::EvalContext; kwargs...) -> EvalContext

Create a new context with additional/overridden bindings.
"""
function with_bindings(ctx::EvalContext, bindings::Dict)
    EvalContext(merge(ctx.bindings, Dict{Symbol,Any}(bindings)), ctx.operators)
end

function with_bindings(ctx::EvalContext; kwargs...)
    d = copy(ctx.bindings)
    for (k, v) in kwargs
        d[k] = v
    end
    EvalContext(d, ctx.operators)
end

# ───────────────────────────────────────────────────────────────────────────────
# Core Evaluation
# ───────────────────────────────────────────────────────────────────────────────

"""
    evaluate(tree::AbstractNode, ctx::EvalContext) -> Any
    evaluate(tree::AbstractNode, grammar::Grammar, ctx::EvalContext) -> Any
    evaluate(tree::AbstractNode; kwargs...) -> Any

Evaluate an expression tree with given variable bindings.

# Examples

```julia
tree = FunctionNode(:+, Variable(:x), Constant(1.0))

# With EvalContext
ctx = EvalContext(x=5.0)
evaluate(tree, ctx)  # 6.0

# With keyword arguments
evaluate(tree, x=5.0)  # 6.0

# With Grammar (uses safe operators)
g = Grammar(binary_operators=[+])
evaluate(tree, g, ctx)  # 6.0
```
"""
function evaluate(tree::AbstractNode, ctx::EvalContext)
    _eval_node(tree, ctx, nothing)
end

function evaluate(tree::AbstractNode, grammar::Grammar, ctx::EvalContext)
    _eval_node(tree, ctx, grammar)
end

function evaluate(tree::AbstractNode; kwargs...)
    ctx = EvalContext(; kwargs...)
    _eval_node(tree, ctx, nothing)
end

function evaluate(tree::AbstractNode, grammar::Grammar; kwargs...)
    ctx = EvalContext(; kwargs...)
    _eval_node(tree, ctx, grammar)
end

# Also support Dict directly
function evaluate(tree::AbstractNode, bindings::Dict)
    ctx = EvalContext(bindings)
    _eval_node(tree, ctx, nothing)
end

function evaluate(tree::AbstractNode, grammar::Grammar, bindings::Dict)
    ctx = EvalContext(bindings)
    _eval_node(tree, ctx, grammar)
end

# ───────────────────────────────────────────────────────────────────────────────
# Internal Evaluation Logic
# ───────────────────────────────────────────────────────────────────────────────

function _eval_node(node::Constant, ctx::EvalContext, grammar)
    return node.value
end

function _eval_node(node::Variable, ctx::EvalContext, grammar)
    if !haskey(ctx, node.name)
        throw(KeyError("Variable '$(node.name)' not found in context. Available: $(keys(ctx))"))
    end
    return ctx[node.name]
end

function _eval_node(node::FunctionNode, ctx::EvalContext, grammar)
    # Special handling for ifelse - LAZY evaluation of branches
    # We only evaluate the branch that's actually taken
    if node.func == :ifelse && length(node.children) == 3
        condition_val = _eval_node(node.children[1], ctx, grammar)
        if !isfinite(condition_val)
            return NaN
        end
        if condition_val >= 0.0
            return _eval_node(node.children[2], ctx, grammar)  # then branch
        else
            return _eval_node(node.children[3], ctx, grammar)  # else branch
        end
    end
    
    # Evaluate children first (eager evaluation for other operators)
    child_values = [_eval_node(child, ctx, grammar) for child in node.children]
    
    # Check for custom context-aware operator first
    if has_operator(ctx, node.func)
        custom_op = get_operator(ctx, node.func)
        try
            # Call with context as last argument
            result = custom_op(child_values..., ctx)
            if result isa Number && !isfinite(result)
                return NaN
            end
            return result
        catch e
            # Fall through to default if custom op fails
            @debug "Custom operator $(node.func) failed: $e"
        end
    end
    
    # Get the function to call from grammar or defaults
    func = _get_eval_function(node.func, length(child_values), grammar)
    
    # Call it
    try
        result = func(child_values...)
        # Validate scalar results, pass through vectors/other types
        if result isa Number && !isfinite(result)
            return NaN
        end
        return result
    catch e
        # Return NaN for numeric errors (only makes sense for scalar results)
        return NaN
    end
end

"""
Get the function to use for evaluation. If grammar is provided, use its safe version.
"""
function _get_eval_function(op::Symbol, arity::Int, grammar::Nothing)
    # No grammar - use default implementations
    return _get_default_function(op, arity)
end

function _get_eval_function(op::Symbol, arity::Int, grammar::Grammar)
    # Try to find in grammar
    ops = operators_by_name(grammar, op)
    for spec in ops
        if spec.arity == arity
            return spec.safe_func
        end
    end
    
    # Fall back to default
    return _get_default_function(op, arity)
end

"""
Default function lookup for common operators.
Uses a module-level constant table to avoid allocating a Dict on every call.
"""
const _DEFAULT_OPERATORS = Dict{Symbol, Function}(
    :+ => +,
    :- => (-),
    :* => *,
    :/ => safe_div,
    :^ => safe_pow,
    :sin => sin,
    :cos => cos,
    :tan => safe_tan,
    :exp => safe_exp,
    :log => safe_log,
    :sqrt => safe_sqrt,
    :abs => abs,
    :neg => (-),
    :inv => safe_inv,
    :max => max,
    :min => min,
    :mean => safe_mean,
    :sum => safe_sum,
    :std => safe_std,
    :var => safe_var,
    :sigmoid => sigmoid,
    :relu => relu,
    :clamp01 => clamp01,
    :tanh => tanh,
)

function _get_default_function(op::Symbol, arity::Int)
    # Check safe implementations first (user-registered overrides)
    if haskey(SAFE_IMPLEMENTATIONS, op)
        return SAFE_IMPLEMENTATIONS[op]
    end
    
    # Constant lookup table — no allocation
    if haskey(_DEFAULT_OPERATORS, op)
        return _DEFAULT_OPERATORS[op]
    end
    
    # Try to find in Base
    if isdefined(Base, op)
        return getfield(Base, op)
    end
    
    error("Unknown operator: $op")
end

# ───────────────────────────────────────────────────────────────────────────────
# Batch Evaluation
# ───────────────────────────────────────────────────────────────────────────────

"""
    evaluate_batch(tree::AbstractNode, data::AbstractMatrix, var_names::Vector{Symbol}) -> Vector
    evaluate_batch(tree::AbstractNode, grammar::Grammar, data::AbstractMatrix, var_names::Vector{Symbol}) -> Vector

Evaluate a tree over multiple data points.

# Arguments
- `tree`: The expression tree to evaluate
- `data`: Matrix where each row is a data point, columns correspond to variables
- `var_names`: Names of variables corresponding to columns

# Returns
Vector of results, one per row of data.

# Example

```julia
tree = FunctionNode(:+, Variable(:x), Variable(:y))
data = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3 rows, 2 columns
results = evaluate_batch(tree, data, [:x, :y])  # [3.0, 7.0, 11.0]
```
"""
function evaluate_batch(tree::AbstractNode, data::AbstractMatrix, var_names::Vector{Symbol})
    n_rows = size(data, 1)
    results = Vector{Float64}(undef, n_rows)
    
    for i in 1:n_rows
        ctx = EvalContext(Dict(var_names[j] => data[i, j] for j in eachindex(var_names)))
        results[i] = evaluate(tree, ctx)
    end
    
    return results
end

function evaluate_batch(tree::AbstractNode, grammar::Grammar, data::AbstractMatrix, var_names::Vector{Symbol})
    n_rows = size(data, 1)
    results = Vector{Float64}(undef, n_rows)
    
    for i in 1:n_rows
        ctx = EvalContext(Dict(var_names[j] => data[i, j] for j in eachindex(var_names)))
        results[i] = evaluate(tree, grammar, ctx)
    end
    
    return results
end

"""
    evaluate_batch(tree::AbstractNode, data::Vector{<:Dict}) -> Vector

Evaluate a tree over a vector of binding dictionaries.
"""
function evaluate_batch(tree::AbstractNode, data::Vector{<:Dict})
    [evaluate(tree, d) for d in data]
end

function evaluate_batch(tree::AbstractNode, grammar::Grammar, data::Vector{<:Dict})
    [evaluate(tree, grammar, d) for d in data]
end

# ───────────────────────────────────────────────────────────────────────────────
# Compiled Evaluation (for performance)
# ───────────────────────────────────────────────────────────────────────────────

"""
    compile_tree(tree::AbstractNode, var_names::Vector{Symbol}) -> Function

Compile an expression tree into a Julia function for faster repeated evaluation.

# Example

```julia
tree = FunctionNode(:*, 
    FunctionNode(:+, Variable(:x), Constant(1.0)),
    Variable(:y)
)

f = compile_tree(tree, [:x, :y])
f(2.0, 3.0)  # (2 + 1) * 3 = 9.0
```
"""
function compile_tree(tree::AbstractNode, var_names::Vector{Symbol})
    # Build the expression
    expr = _tree_to_expr(tree)
    
    # Create argument symbols
    arg_symbols = [Symbol("_arg_$i") for i in eachindex(var_names)]
    
    # Create let bindings to map var names to argument positions
    let_bindings = [:($(var_names[i]) = $(arg_symbols[i])) for i in eachindex(var_names)]
    
    # Build the let block
    let_block = Expr(:let, Expr(:block, let_bindings...), expr)
    
    # Build the function expression using Expr directly
    func_expr = Expr(:->, Expr(:tuple, arg_symbols...), let_block)
    
    return eval(func_expr)
end

"""
Convert a tree to a Julia expression.
"""
function _tree_to_expr(node::Constant)
    return node.value
end

function _tree_to_expr(node::Variable)
    return node.name
end

function _tree_to_expr(node::FunctionNode)
    child_exprs = [_tree_to_expr(child) for child in node.children]
    
    # Handle infix operators specially
    if node.func in [:+, :-, :*, :/, :^] && length(child_exprs) == 2
        return Expr(:call, node.func, child_exprs...)
    else
        # Get the safe function if available
        func_sym = _get_safe_func_symbol(node.func)
        return Expr(:call, func_sym, child_exprs...)
    end
end

function _get_safe_func_symbol(op::Symbol)
    safe_map = Dict(
        :/ => :safe_div,
        :^ => :safe_pow,
        :log => :safe_log,
        :exp => :safe_exp,
        :sqrt => :safe_sqrt,
        :tan => :safe_tan,
        :mean => :safe_mean,
        :sum => :safe_sum,
        :std => :safe_std,
        :var => :safe_var,
    )
    return get(safe_map, op, op)
end

# ───────────────────────────────────────────────────────────────────────────────
# Evaluation with Error Handling
# ───────────────────────────────────────────────────────────────────────────────

"""
    safe_evaluate(tree::AbstractNode, ctx::EvalContext) -> Float64

Evaluate a tree, returning NaN for any errors.
"""
function safe_evaluate(tree::AbstractNode, ctx::EvalContext)
    try
        result = evaluate(tree, ctx)
        return isfinite(result) ? Float64(result) : NaN
    catch
        return NaN
    end
end

function safe_evaluate(tree::AbstractNode, grammar::Grammar, ctx::EvalContext)
    try
        result = evaluate(tree, grammar, ctx)
        return isfinite(result) ? Float64(result) : NaN
    catch
        return NaN
    end
end

"""
    is_valid_on(tree::AbstractNode, ctx::EvalContext) -> Bool

Check if a tree evaluates to a finite value for the given context.
"""
function is_valid_on(tree::AbstractNode, ctx::EvalContext)
    result = safe_evaluate(tree, ctx)
    return isfinite(result)
end

function is_valid_on(tree::AbstractNode, grammar::Grammar, ctx::EvalContext)
    result = safe_evaluate(tree, grammar, ctx)
    return isfinite(result)
end
