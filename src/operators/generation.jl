# ═══════════════════════════════════════════════════════════════════════════════
# Tree Generation
# ═══════════════════════════════════════════════════════════════════════════════
#
# Functions for generating random expression trees from a grammar.

using Random

# ───────────────────────────────────────────────────────────────────────────────
# Generation Methods
# ───────────────────────────────────────────────────────────────────────────────

"""
    GenerationMethod

Abstract type for tree generation strategies.
"""
abstract type GenerationMethod end

"""
    FullMethod <: GenerationMethod

Generate trees where all branches have exactly the specified depth.
All paths from root to leaves have the same length.
"""
struct FullMethod <: GenerationMethod end

"""
    GrowMethod <: GenerationMethod

Generate trees where branches can terminate early.
Produces more varied tree shapes.
"""
struct GrowMethod <: GenerationMethod end

"""
    HalfAndHalfMethod <: GenerationMethod

Generate half the trees with Full method, half with Grow method.
Provides good diversity in initial populations.
"""
struct HalfAndHalfMethod <: GenerationMethod end

# ───────────────────────────────────────────────────────────────────────────────
# Main Generation Functions
# ───────────────────────────────────────────────────────────────────────────────

"""
    generate_tree(grammar::Grammar; kwargs...) -> AbstractNode

Generate a random expression tree from the grammar.

# Keyword Arguments
- `method::GenerationMethod = GrowMethod()`: Generation strategy
- `min_depth::Int = 1`: Minimum tree depth
- `max_depth::Int = 4`: Maximum tree depth
- `target_type::Symbol = :Any`: Target output type (for typed grammars)
- `rng::AbstractRNG = Random.GLOBAL_RNG`: Random number generator

# Examples

```julia
g = Grammar(binary_operators=[+, *], unary_operators=[sin], variables=[:x])

tree = generate_tree(g, max_depth=3)
tree = generate_tree(g, method=FullMethod(), max_depth=4)
tree = generate_tree(g, method=HalfAndHalfMethod(), min_depth=2, max_depth=5)
```
"""
function generate_tree(grammar::Grammar;
                       method::GenerationMethod = GrowMethod(),
                       min_depth::Int = 1,
                       max_depth::Int = 4,
                       target_type::Symbol = :Any,
                       rng::AbstractRNG = Random.GLOBAL_RNG)
    
    if method isa HalfAndHalfMethod
        # Randomly choose between Full and Grow
        actual_method = rand(rng, Bool) ? FullMethod() : GrowMethod()
        return generate_tree(grammar; method=actual_method, min_depth, max_depth, target_type, rng)
    end
    
    # Determine actual target type
    actual_target = if target_type == :Any && grammar.is_typed && !isempty(grammar.output_types)
        rand(rng, collect(grammar.output_types))
    else
        target_type
    end
    
    return _generate_node(grammar, method, 1, min_depth, max_depth, actual_target, rng)
end

"""
    generate_population(grammar::Grammar, n::Int; kwargs...) -> Vector{AbstractNode}

Generate a population of random trees.

# Arguments
- `grammar`: The grammar to use
- `n`: Number of trees to generate

# Keyword Arguments
Same as `generate_tree`, plus:
- `unique::Bool = false`: If true, ensure all trees are structurally unique
- `max_attempts::Int = n * 10`: Maximum attempts when generating unique trees
"""
function generate_population(grammar::Grammar, n::Int;
                            method::GenerationMethod = HalfAndHalfMethod(),
                            min_depth::Int = 2,
                            max_depth::Int = 5,
                            target_type::Symbol = :Any,
                            unique::Bool = false,
                            max_attempts::Int = n * 10,
                            rng::AbstractRNG = Random.GLOBAL_RNG)
    
    trees = AbstractNode[]
    seen = unique ? Set{UInt64}() : nothing
    attempts = 0
    
    while length(trees) < n && attempts < max_attempts
        attempts += 1
        
        # Vary depth for diversity
        depth = rand(rng, min_depth:max_depth)
        
        tree = generate_tree(grammar;
                            method = method,
                            min_depth = min_depth,
                            max_depth = depth,
                            target_type = target_type,
                            rng = rng)
        
        if unique
            h = hash(tree)
            if h ∉ seen
                push!(seen, h)
                push!(trees, tree)
            end
        else
            push!(trees, tree)
        end
    end
    
    if length(trees) < n
        @warn "Could only generate $(length(trees)) unique trees (requested $n)"
    end
    
    return trees
end

# ───────────────────────────────────────────────────────────────────────────────
# Internal Generation Logic
# ───────────────────────────────────────────────────────────────────────────────

function _generate_node(grammar::Grammar, method::GenerationMethod,
                        current_depth::Int, min_depth::Int, max_depth::Int,
                        target_type::Symbol, rng::AbstractRNG)
    
    # At max depth, must generate terminal
    if current_depth >= max_depth
        return _generate_terminal(grammar, target_type, rng)
    end
    
    # Below min depth, must generate function (if possible)
    if current_depth < min_depth
        return _generate_function_node(grammar, method, current_depth, min_depth, max_depth, target_type, rng)
    end
    
    # Between min and max depth: depends on method
    if method isa FullMethod
        # Full: always expand until max depth
        return _generate_function_node(grammar, method, current_depth, min_depth, max_depth, target_type, rng)
    else
        # Grow: randomly choose terminal or function
        # Bias towards functions at shallow depths
        terminal_prob = current_depth / max_depth * 0.5
        
        if rand(rng) < terminal_prob
            return _generate_terminal(grammar, target_type, rng)
        else
            node = _try_generate_function_node(grammar, method, current_depth, min_depth, max_depth, target_type, rng)
            return node !== nothing ? node : _generate_terminal(grammar, target_type, rng)
        end
    end
end

function _generate_terminal(grammar::Grammar, target_type::Symbol, rng::AbstractRNG)
    # Collect valid terminals for the target type
    valid_vars = if target_type == :Any || !grammar.is_typed
        grammar.variables
    else
        vars = variables_of_type(grammar, target_type)
        isempty(vars) ? grammar.variables : vars
    end
    
    valid_consts = if target_type == :Any || !grammar.is_typed
        grammar.constants
    else
        consts = constants_of_type(grammar, target_type)
        isempty(consts) ? grammar.constants : consts
    end
    
    n_vars = length(valid_vars)
    n_consts = length(valid_consts)
    
    # Handle edge cases
    if n_vars == 0 && n_consts == 0
        # Fallback: create a constant
        return Constant(0.0)
    elseif n_consts == 0
        # No constants available, must use variable
        var_spec = rand(rng, valid_vars)
        if grammar.is_typed && var_spec.type != :Any
            return Variable(var_spec.name, var_spec.type)
        else
            return Variable(var_spec.name)
        end
    elseif n_vars == 0
        # No variables available, must use constant
        const_spec = rand(rng, valid_consts)
        return Constant(const_spec.sampler())
    end
    
    # Use grammar's constant_prob to decide between constant and variable
    # (This is independent of the number of variables/constants available)
    if rand(rng) < grammar.constant_prob
        # Generate a constant
        const_spec = rand(rng, valid_consts)
        return Constant(const_spec.sampler())
    else
        # Choose a variable
        var_spec = rand(rng, valid_vars)
        if grammar.is_typed && var_spec.type != :Any
            return Variable(var_spec.name, var_spec.type)
        else
            return Variable(var_spec.name)
        end
    end
end

function _generate_function_node(grammar::Grammar, method::GenerationMethod,
                                  current_depth::Int, min_depth::Int, max_depth::Int,
                                  target_type::Symbol, rng::AbstractRNG)
    node = _try_generate_function_node(grammar, method, current_depth, min_depth, max_depth, target_type, rng)
    if node !== nothing
        return node
    end
    # Fallback to terminal if no valid operator found
    return _generate_terminal(grammar, target_type, rng)
end

function _try_generate_function_node(grammar::Grammar, method::GenerationMethod,
                                      current_depth::Int, min_depth::Int, max_depth::Int,
                                      target_type::Symbol, rng::AbstractRNG)
    # Get operators that produce the target type
    valid_ops = if target_type == :Any || !grammar.is_typed
        grammar.operators
    else
        ops = operators_producing(grammar, target_type)
        isempty(ops) ? grammar.operators : ops
    end
    
    if isempty(valid_ops)
        return nothing
    end
    
    # Shuffle and try operators
    shuffled_ops = shuffle(rng, valid_ops)
    
    for op in shuffled_ops
        children = AbstractNode[]
        success = true
        
        for (i, input_type) in enumerate(op.input_types)
            child_type = isempty(op.input_types) ? :Any : input_type
            child = _generate_node(grammar, method, current_depth + 1, min_depth, max_depth, child_type, rng)
            push!(children, child)
        end
        
        # For untyped grammars with arity but no input types
        if isempty(op.input_types)
            for _ in 1:op.arity
                child = _generate_node(grammar, method, current_depth + 1, min_depth, max_depth, :Any, rng)
                push!(children, child)
            end
        end
        
        if success && length(children) == op.arity
            return FunctionNode(op.name, children)
        end
    end
    
    return nothing
end

# ───────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ───────────────────────────────────────────────────────────────────────────────

"""
    random_terminal(grammar::Grammar; target_type=:Any, rng=Random.GLOBAL_RNG) -> AbstractNode

Generate a random terminal node (variable or constant).
"""
function random_terminal(grammar::Grammar; target_type::Symbol=:Any, rng::AbstractRNG=Random.GLOBAL_RNG)
    _generate_terminal(grammar, target_type, rng)
end

"""
    generate_random_subtree(grammar::Grammar; max_depth=3, target_type=:Any, rng=Random.GLOBAL_RNG) -> AbstractNode

Generate a random subtree with the given maximum depth.
Alias for `generate_tree` with `GrowMethod`.
"""
function generate_random_subtree(grammar::Grammar; 
                                  max_depth::Int=3, 
                                  target_type::Symbol=:Any, 
                                  rng::AbstractRNG=Random.GLOBAL_RNG)
    generate_tree(grammar; method=GrowMethod(), min_depth=1, max_depth=max_depth, target_type=target_type, rng=rng)
end
