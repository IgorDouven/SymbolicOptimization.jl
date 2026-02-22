# ═══════════════════════════════════════════════════════════════════════════════
# Tree Simplification
# ═══════════════════════════════════════════════════════════════════════════════
#
# Functions for simplifying expression trees.

# ───────────────────────────────────────────────────────────────────────────────
# Constant Folding
# ───────────────────────────────────────────────────────────────────────────────

"""
    simplify_constants(tree::AbstractNode; grammar::Union{Grammar,Nothing}=nothing) -> AbstractNode

Evaluate and fold constant subexpressions.

Replaces subtrees that contain only constants with their evaluated result.

Note: This function only folds existing constants - it does not create new ones
from variable expressions. Use `simplify_algebra` for algebraic simplifications.
"""
function simplify_constants(tree::AbstractNode; grammar::Union{Grammar,Nothing}=nothing)
    # Determine if constants are allowed based on grammar
    allow_constants = grammar === nothing || !isempty(grammar.constants)
    _fold_constants(tree, grammar, allow_constants)
end

function _fold_constants(node::Constant, grammar, allow_constants::Bool)
    return node
end

function _fold_constants(node::Variable, grammar, allow_constants::Bool)
    return node
end

function _fold_constants(node::FunctionNode, grammar, allow_constants::Bool)
    # First, recursively simplify children
    new_children = AbstractNode[_fold_constants(c, grammar, allow_constants) for c in node.children]
    
    # Check if all children are constants AND we're allowed to create constants
    if allow_constants && all(c -> c isa Constant, new_children)
        # Evaluate this subtree
        try
            # Build minimal context (no variables needed)
            ctx = EvalContext()
            simplified_node = FunctionNode(node.func, new_children)
            
            if grammar !== nothing
                result = evaluate(simplified_node, grammar, ctx)
            else
                result = evaluate(simplified_node, ctx)
            end
            
            if isfinite(result)
                return Constant(result)
            end
        catch
            # If evaluation fails, keep the expression
        end
    end
    
    return FunctionNode(node.func, new_children)
end

# ───────────────────────────────────────────────────────────────────────────────
# Algebraic Simplification
# ───────────────────────────────────────────────────────────────────────────────

"""
    simplify_algebra(tree::AbstractNode; grammar::Union{Grammar,Nothing}=nothing) -> AbstractNode

Apply basic algebraic simplification rules.

Rules include:
- `x + 0 → x`, `x - 0 → x`, `x * 1 → x`, `x / 1 → x`
- `x * 0 → 0`, `0 / x → 0` (only if constants allowed)
- `x - x → 0`, `x / x → 1` (only if constants allowed)
- `x ^ 0 → 1`, `x ^ 1 → x` (constant creation only if allowed)
- `sin(0) → 0`, `cos(0) → 1`, etc. (only if constants allowed)

When `grammar` is provided and has no constants (`isempty(grammar.constants)`),
rules that would create new constant nodes are skipped. This ensures that
`constant_range = nothing` is respected throughout the optimization process.
"""
function simplify_algebra(tree::AbstractNode; grammar::Union{Grammar,Nothing}=nothing)
    # Determine if constants are allowed based on grammar
    allow_constants = grammar === nothing || !isempty(grammar.constants)
    _simplify_algebraic(tree, allow_constants)
end

function _simplify_algebraic(node::Constant, allow_constants::Bool)
    return node
end

function _simplify_algebraic(node::Variable, allow_constants::Bool)
    return node
end

function _simplify_algebraic(node::FunctionNode, allow_constants::Bool)
    # First simplify children
    new_children = AbstractNode[_simplify_algebraic(c, allow_constants) for c in node.children]
    
    # Apply simplification rules based on the function
    result = _apply_rules(node.func, new_children, allow_constants)
    
    return result
end

function _apply_rules(func::Symbol, children::Vector{AbstractNode}, allow_constants::Bool)
    n = length(children)
    
    # Binary operators
    if n == 2
        a, b = children
        
        # Addition rules
        if func == :+
            # These rules don't create new constants, they just return existing nodes
            if _is_zero(a)
                return b
            elseif _is_zero(b)
                return a
            # x + x → 2*x creates a new constant, so check allow_constants
            elseif _are_equal(a, b) && allow_constants
                return FunctionNode(:*, Constant(2.0), a)
            end
        
        # Subtraction rules
        elseif func == :-
            # x - 0 → x doesn't create a constant
            if _is_zero(b)
                return a
            # x - x → 0 creates a constant
            elseif _are_equal(a, b) && allow_constants
                return Constant(0.0)
            end
        
        # Multiplication rules
        elseif func == :*
            # x * 0 → 0 creates a constant (unless 0 is already there)
            if (_is_zero(a) || _is_zero(b)) && allow_constants
                return Constant(0.0)
            # x * 1 → x doesn't create a constant
            elseif _is_one(a)
                return b
            elseif _is_one(b)
                return a
            end
        
        # Division rules
        elseif func == :/
            # 0 / x → 0 creates a constant
            if _is_zero(a) && allow_constants
                return Constant(0.0)
            # x / 1 → x doesn't create a constant
            elseif _is_one(b)
                return a
            # x / x → 1 creates a constant
            elseif _are_equal(a, b) && allow_constants
                return Constant(1.0)
            end
        
        # Power rules
        elseif func == :^
            # x^0 → 1 creates a constant
            if _is_zero(b) && allow_constants
                return Constant(1.0)
            # x^1 → x doesn't create a constant
            elseif _is_one(b)
                return a
            # 0^x → 0 creates a constant
            elseif _is_zero(a) && allow_constants
                return Constant(0.0)
            # 1^x → 1 creates a constant
            elseif _is_one(a) && allow_constants
                return Constant(1.0)
            end
        end
    
    # Unary operators
    elseif n == 1
        a = children[1]
        
        # All of these create new constants, so check allow_constants
        if allow_constants
            if func == :sin && _is_zero(a)
                return Constant(0.0)
            elseif func == :cos && _is_zero(a)
                return Constant(1.0)
            elseif func == :exp && _is_zero(a)
                return Constant(1.0)
            elseif func == :log && _is_one(a)
                return Constant(0.0)
            elseif func == :sqrt && _is_zero(a)
                return Constant(0.0)
            elseif func == :sqrt && _is_one(a)
                return Constant(1.0)
            elseif func == :neg && a isa Constant
                return Constant(-a.value)
            end
        end
        
        # This rule doesn't create a new constant, just returns existing one
        if func == :abs && a isa Constant && a.value >= 0
            return a
        end
    end
    
    # No simplification applied
    return FunctionNode(func, children)
end

# Helper functions for rule matching
_is_zero(node::Constant) = abs(node.value) < 1e-10
_is_zero(node::AbstractNode) = false

_is_one(node::Constant) = abs(node.value - 1.0) < 1e-10
_is_one(node::AbstractNode) = false

function _are_equal(a::AbstractNode, b::AbstractNode)
    if a isa Constant && b isa Constant
        return abs(a.value - b.value) < 1e-10
    elseif a isa Variable && b isa Variable
        return a.name == b.name
    elseif a isa FunctionNode && b isa FunctionNode
        if a.func != b.func || length(a.children) != length(b.children)
            return false
        end
        return all(_are_equal(ac, bc) for (ac, bc) in zip(a.children, b.children))
    end
    return false
end

# ───────────────────────────────────────────────────────────────────────────────
# Combined Simplification
# ───────────────────────────────────────────────────────────────────────────────

"""
    simplify(tree::AbstractNode; grammar::Union{Grammar,Nothing}=nothing, iterations::Int=3) -> AbstractNode

Apply all simplification passes repeatedly until the tree stabilizes.

# Arguments
- `tree`: The tree to simplify
- `grammar`: Optional grammar for evaluation during constant folding.
             If provided with `isempty(grammar.constants)`, no new constants
             will be created during simplification.
- `iterations`: Maximum number of simplification passes
"""
function simplify(tree::AbstractNode; 
                  grammar::Union{Grammar,Nothing}=nothing, 
                  iterations::Int=3)
    current = tree
    
    for _ in 1:iterations
        # Apply simplifications - pass grammar to respect constant settings
        simplified = simplify_algebra(current; grammar=grammar)
        simplified = simplify_constants(simplified; grammar=grammar)
        
        # Check if anything changed
        if hash(simplified) == hash(current)
            break
        end
        
        current = simplified
    end
    
    return current
end

# ───────────────────────────────────────────────────────────────────────────────
# Tree Normalization
# ───────────────────────────────────────────────────────────────────────────────

"""
    normalize_constants(tree::AbstractNode; digits::Int=6) -> AbstractNode

Round all constants to a specified number of digits.
Helps with numerical stability and canonical forms.
"""
function normalize_constants(tree::AbstractNode; digits::Int=6)
    map_tree(tree) do node
        if node isa Constant
            Constant(round(node.value; digits=digits))
        else
            node
        end
    end
end

"""
    clamp_constants(tree::AbstractNode; min_val::Float64=-1e6, max_val::Float64=1e6) -> AbstractNode

Clamp all constants to a range to prevent extreme values.
"""
function clamp_constants(tree::AbstractNode; min_val::Float64=-1e6, max_val::Float64=1e6)
    map_tree(tree) do node
        if node isa Constant
            Constant(clamp(node.value, min_val, max_val))
        else
            node
        end
    end
end
