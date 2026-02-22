"""
    SymbolicsExt

Package extension that provides deep algebraic simplification via Symbolics.jl.
Loaded automatically when the user does `using Symbolics` alongside SymbolicOptimization.
"""
module SymbolicsExt

using SymbolicOptimization
using Symbolics
import SymbolicUtils

# ═══════════════════════════════════════════════════════════════════════════════
# AbstractNode → Symbolics.Num
# ═══════════════════════════════════════════════════════════════════════════════

# Map from safe operator names to the standard functions Symbolics understands.
const _SAFE_TO_STANDARD = Dict{Symbol, Any}(
    :safe_div   => /,
    :safe_pow   => ^,
    :safe_log   => log,
    :safe_log10 => log10,
    :safe_log2  => log2,
    :safe_exp   => exp,
    :safe_sqrt  => sqrt,
    :safe_inv   => inv,
    :safe_tan   => tan,
    :safe_asin  => asin,
    :safe_acos  => acos,
    :safe_sinh  => sinh,
    :safe_cosh  => cosh,
    :safe_tanh  => tanh,
    :safe_acosh => acosh,
    :safe_atanh => atanh,
    :safe_abs   => abs,
    :safe_sign  => sign,
    :safe_max   => max,
    :safe_min   => min,
    :neg        => (-),
    :inv        => inv,
)

# Standard operators that Symbolics knows natively.
const _STANDARD_OPS = Dict{Symbol, Any}(
    :+    => +,
    :-    => -,
    :*    => *,
    :/    => /,
    :^    => ^,
    :sin  => sin,
    :cos  => cos,
    :tan  => tan,
    :exp  => exp,
    :log  => log,
    :log10 => log10,
    :log2  => log2,
    :sqrt => sqrt,
    :abs  => abs,
    :sign => sign,
    :asin => asin,
    :acos => acos,
    :atan => atan,
    :sinh => sinh,
    :cosh => cosh,
    :tanh => tanh,
    :max  => max,
    :min  => min,
)

"""
    _make_sym_var(name::Symbol, cache::Dict{Symbol, Symbolics.Num}) -> Symbolics.Num

Get or create a Symbolics variable for the given name.
"""
function _make_sym_var(name::Symbol, cache::Dict{Symbol, Symbolics.Num})
    if haskey(cache, name)
        return cache[name]
    end
    sv = Symbolics.variable(name)
    cache[name] = sv
    return sv
end

"""
    _to_sym(node::AbstractNode, var_cache::Dict{Symbol, Symbolics.Num})

Internal recursive conversion from AbstractNode to Symbolics expression.
Returns a Symbolics.Num or a plain number.
"""
function _to_sym(node::SymbolicOptimization.Constant, vc::Dict{Symbol, Symbolics.Num})
    return node.value
end

function _to_sym(node::SymbolicOptimization.Variable, vc::Dict{Symbol, Symbolics.Num})
    return _make_sym_var(node.name, vc)
end

function _to_sym(node::SymbolicOptimization.FunctionNode, vc::Dict{Symbol, Symbolics.Num})
    args = [_to_sym(c, vc) for c in SymbolicOptimization.children(node)]
    op = node.func

    # Look up the function to apply
    f = get(_SAFE_TO_STANDARD, op, get(_STANDARD_OPS, op, nothing))

    if f !== nothing
        # Handle unary minus
        if f === (-) && length(args) == 1
            return -args[1]
        end
        return f(args...)
    end

    # Fallback: try to find in Base
    if isdefined(Base, op)
        f = getfield(Base, op)
        return f(args...)
    end

    # Last resort: check if it's a known indicator/piecewise function
    _INDICATOR_OPS = Set([:step_func, :step, :heaviside, :indicator])
    if op in _INDICATOR_OPS
        @warn "Operator :$op is a piecewise indicator. Use `simplify_piecewise(tree)` instead of `deep_simplify` for better results." maxlog=1
    else
        @warn "Operator :$op is not known to Symbolics; it will be left as-is." maxlog=1
    end
    # Return a placeholder variable so the rest of the expression can still simplify
    placeholder = Symbolics.variable(Symbol("_$(op)_", join(["arg$i" for i in 1:length(args)], "_")))
    return placeholder
end

function SymbolicOptimization.to_symbolics(tree::SymbolicOptimization.AbstractNode)
    vc = Dict{Symbol, Symbolics.Num}()
    return _to_sym(tree, vc)
end


# ═══════════════════════════════════════════════════════════════════════════════
# Symbolics.Num → AbstractNode
# ═══════════════════════════════════════════════════════════════════════════════

# Reverse map: Julia function → tree operator symbol
const _FUNC_TO_SYMBOL = Dict{Any, Symbol}(
    (+)    => :+,
    (-)    => :-,
    (*)    => :*,
    (/)    => :/,
    (^)    => :^,
    sin    => :sin,
    cos    => :cos,
    tan    => :tan,
    exp    => :exp,
    log    => :log,
    log10  => :log10,
    log2   => :log2,
    sqrt   => :sqrt,
    abs    => :abs,
    sign   => :sign,
    asin   => :asin,
    acos   => :acos,
    atan   => :atan,
    sinh   => :sinh,
    cosh   => :cosh,
    tanh   => :tanh,
    max    => :max,
    min    => :min,
    inv    => :inv,
)

"""
    _from_sym(expr) -> AbstractNode

Recursively convert a Symbolics/SymbolicUtils expression back to an AbstractNode tree.
"""
function _from_sym(x::Real)
    # Plain numeric values (Int, Float64, etc. — but NOT Symbolics.Num)
    return SymbolicOptimization.Constant(Float64(x))
end

function _from_sym(x::Symbolics.Num)
    # Unwrap the Num wrapper and recurse on the inner symbolic expression
    u = Symbolics.unwrap(x)
    return _from_sym_inner(u)
end

function _from_sym(x)
    # Generic fallback: try unwrapping in case it's some other wrapper
    u = Symbolics.unwrap(x)
    return _from_sym_inner(u)
end

function _from_sym_inner(x::Real)
    return SymbolicOptimization.Constant(Float64(x))
end

function _from_sym_inner(x)
    # Check if it's a symbolic variable (leaf)
    if SymbolicUtils.issym(x)
        name = SymbolicUtils.nameof(x)
        return SymbolicOptimization.Variable(name)
    end

    # Check if it's a function call
    if SymbolicUtils.iscall(x)
        op_func = SymbolicUtils.operation(x)
        raw_args = SymbolicUtils.arguments(x)
        child_nodes = SymbolicOptimization.AbstractNode[_from_sym(a) for a in raw_args]

        # Map function back to symbol
        op_sym = get(_FUNC_TO_SYMBOL, op_func, nothing)

        if op_sym !== nothing
            # Special case: binary minus with one argument means negation
            if op_sym == :- && length(child_nodes) == 1
                return SymbolicOptimization.FunctionNode(:neg, child_nodes)
            end
            return SymbolicOptimization.FunctionNode(op_sym, child_nodes)
        end

        # Unknown function — try to recover the name
        op_name = try
            Symbol(op_func)
        catch
            nameof(op_func)
        end
        return SymbolicOptimization.FunctionNode(op_name, child_nodes)
    end

    # If we get here, try treating as a numeric constant
    if x isa Real
        return SymbolicOptimization.Constant(Float64(x))
    end
    error("Cannot convert Symbolics expression to AbstractNode: $x (type: $(typeof(x)))")
end

function SymbolicOptimization.from_symbolics(expr)
    return _from_sym(expr)
end


# ═══════════════════════════════════════════════════════════════════════════════
# High-Level Functions
# ═══════════════════════════════════════════════════════════════════════════════

function SymbolicOptimization.deep_simplify(tree::SymbolicOptimization.AbstractNode; 
                                             expand::Bool=false)
    sym_expr = SymbolicOptimization.to_symbolics(tree)
    simplified = Symbolics.simplify(sym_expr; expand=expand)
    return SymbolicOptimization.from_symbolics(simplified)
end

function SymbolicOptimization.simplified_string(tree::SymbolicOptimization.AbstractNode; 
                                                 digits::Int=3)
    sym_expr = SymbolicOptimization.to_symbolics(tree)
    simplified = Symbolics.simplify(sym_expr)
    s = string(simplified)
    # Round numeric literals in the string for cleaner output
    return s
end

function SymbolicOptimization.simplified_latex(tree::SymbolicOptimization.AbstractNode)
    sym_expr = SymbolicOptimization.to_symbolics(tree)
    simplified = Symbolics.simplify(sym_expr)
    # Use the LaTeX MIME show method that Symbolics provides
    io = IOBuffer()
    show(io, MIME("text/latex"), simplified)
    s = String(take!(io))
    # Strip any $...$ or $$...$$ delimiters 
    s = replace(s, r"^\${1,2}" => "")
    s = replace(s, r"\${1,2}$" => "")
    return strip(s)
end


# ═══════════════════════════════════════════════════════════════════════════════
# Piecewise Simplification
# ═══════════════════════════════════════════════════════════════════════════════

"""
Find all nodes matching `indicator(cond)` and return the condition subtree.
Assumes all occurrences share the same condition.
"""
function _find_indicator_condition(tree::SymbolicOptimization.AbstractNode, 
                                   indicator::Symbol)
    if tree isa SymbolicOptimization.FunctionNode
        if tree.func == indicator && length(SymbolicOptimization.children(tree)) == 1
            return SymbolicOptimization.children(tree)[1]
        end
        for child in SymbolicOptimization.children(tree)
            result = _find_indicator_condition(child, indicator)
            if result !== nothing
                return result
            end
        end
    end
    return nothing
end

"""
Replace all occurrences of `indicator(...)` with `replacement` in the tree.
"""
function _substitute_indicator(tree::SymbolicOptimization.AbstractNode, 
                                indicator::Symbol,
                                replacement::SymbolicOptimization.AbstractNode)
    if tree isa SymbolicOptimization.FunctionNode
        if tree.func == indicator && length(SymbolicOptimization.children(tree)) == 1
            return SymbolicOptimization.copy_tree(replacement)
        end
        new_children = SymbolicOptimization.AbstractNode[
            _substitute_indicator(c, indicator, replacement) 
            for c in SymbolicOptimization.children(tree)
        ]
        return SymbolicOptimization.FunctionNode(tree.func, new_children)
    else
        return SymbolicOptimization.copy_tree(tree)
    end
end

"""
Replace variables in `tree` according to a substitution dictionary.
Each key is a variable name to replace; the value is the replacement subtree.
"""
function _apply_substitutions(tree::SymbolicOptimization.AbstractNode,
                               subs::Dict{Symbol, <:SymbolicOptimization.AbstractNode})
    if tree isa SymbolicOptimization.Variable
        if haskey(subs, tree.name)
            return SymbolicOptimization.copy_tree(subs[tree.name])
        else
            return SymbolicOptimization.copy_tree(tree)
        end
    elseif tree isa SymbolicOptimization.FunctionNode
        new_children = SymbolicOptimization.AbstractNode[
            _apply_substitutions(c, subs) 
            for c in SymbolicOptimization.children(tree)
        ]
        return SymbolicOptimization.FunctionNode(tree.func, new_children)
    else
        return SymbolicOptimization.copy_tree(tree)
    end
end

function SymbolicOptimization.simplify_piecewise(
        tree::SymbolicOptimization.AbstractNode; 
        indicator::Symbol=:step_func,
        substitutions::Dict{Symbol, <:SymbolicOptimization.AbstractNode}=Dict{Symbol, SymbolicOptimization.AbstractNode}())
    
    # 1. Find the condition inside step_func(cond)
    condition = _find_indicator_condition(tree, indicator)
    if condition === nothing
        error("No $indicator(...) found in tree. Use deep_simplify instead for non-piecewise formulas.")
    end
    
    # 2. Substitute indicator → 1 (if-branch) and indicator → 0 (else-branch)
    if_tree = _substitute_indicator(tree, indicator, SymbolicOptimization.Constant(1.0))
    else_tree = _substitute_indicator(tree, indicator, SymbolicOptimization.Constant(0.0))
    
    # 3. Apply built-in simplify first (folds 0*x → 0, 1*x → x, etc.)
    if_tree = SymbolicOptimization.simplify(if_tree)
    else_tree = SymbolicOptimization.simplify(else_tree)
    
    # 4. Convert to Symbolics space
    if_sym = SymbolicOptimization.to_symbolics(if_tree)
    else_sym = SymbolicOptimization.to_symbolics(else_tree)
    
    # 5. Apply domain substitutions in Symbolics space (key for proper simplification)
    if !isempty(substitutions)
        sym_subs = Dict{Any, Any}()
        for (name, repl_tree) in substitutions
            sym_var = Symbolics.variable(name)
            # Convert the replacement tree to a Symbolics expression
            repl_vc = Dict{Symbol, Symbolics.Num}()
            sym_repl = _to_sym(repl_tree, repl_vc)
            sym_subs[sym_var] = sym_repl
        end
        if_sym = Symbolics.substitute(if_sym, sym_subs)
        else_sym = Symbolics.substitute(else_sym, sym_subs)
    end
    
    # 6. Simplify in Symbolics (expand then re-simplify for best factoring)
    if_simplified_sym = Symbolics.simplify(if_sym)
    else_simplified_sym = Symbolics.simplify(else_sym)
    
    # 7. Convert back to AbstractNode
    if_simplified = SymbolicOptimization.from_symbolics(if_simplified_sym)
    else_simplified = SymbolicOptimization.from_symbolics(else_simplified_sym)
    
    # 8. Generate string and LaTeX representations
    cond_str = SymbolicOptimization.node_to_string(condition)
    if_str = string(if_simplified_sym)
    else_str = string(else_simplified_sym)
    
    if_io = IOBuffer()
    show(if_io, MIME("text/latex"), if_simplified_sym)
    if_latex = String(take!(if_io))
    if_latex = replace(if_latex, r"^\${1,2}" => "")
    if_latex = replace(if_latex, r"\${1,2}$" => "")
    if_latex = strip(if_latex)
    
    else_io = IOBuffer()
    show(else_io, MIME("text/latex"), else_simplified_sym)
    else_latex = String(take!(else_io))
    else_latex = replace(else_latex, r"^\${1,2}" => "")
    else_latex = replace(else_latex, r"\${1,2}$" => "")
    else_latex = strip(else_latex)
    
    return SymbolicOptimization.PiecewiseResult(
        condition, cond_str,
        if_simplified, else_simplified,
        if_str, else_str,
        String(if_latex), String(else_latex)
    )
end

end # module
