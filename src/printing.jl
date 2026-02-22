# ═══════════════════════════════════════════════════════════════════════════════
# Tree Printing and String Conversion
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
# Infix operators for pretty printing
# ───────────────────────────────────────────────────────────────────────────────

const INFIX_OPERATORS = Set([:+, :-, :*, :/, :^, :.+, :.-, :.*, :./, :.^])

const OPERATOR_PRECEDENCE = Dict(
    :+ => 1, :- => 1, :.+ => 1, :.- => 1,
    :* => 2, :/ => 2, :.* => 2, :./ => 2,
    :^ => 3, :.^ => 3,
)

# ───────────────────────────────────────────────────────────────────────────────
# String Representation
# ───────────────────────────────────────────────────────────────────────────────

"""
    node_to_string(node::AbstractNode; digits::Int=3) -> String

Convert an expression tree to a human-readable string representation.

Infix operators (+, -, *, /, ^) are printed in infix notation.
Other functions use prefix notation: `f(arg1, arg2, ...)`.

# Examples
```julia
tree = FunctionNode(:+, Variable(:x), FunctionNode(:*, Constant(2.0), Variable(:y)))
node_to_string(tree)  # "(x + (2.0 * y))"

tree2 = FunctionNode(:sin, Variable(:x))
node_to_string(tree2)  # "sin(x)"
```
"""
function node_to_string(node::AbstractNode; digits::Int=3)
    return _node_to_string(node, digits, 0)
end

function _node_to_string(node::Constant, digits::Int, parent_prec::Int)
    if node.value == round(node.value) && abs(node.value) < 1e6
        return string(Int(node.value))
    else
        return string(round(node.value; digits=digits))
    end
end

function _node_to_string(node::Variable, digits::Int, parent_prec::Int)
    return string(node.name)
end

function _node_to_string(node::FunctionNode, digits::Int, parent_prec::Int)
    func = node.func
    args = node.children
    
    # Special handling for ifelse - use mathematical notation
    if func == :ifelse && length(args) == 3
        cond = _node_to_string(args[1], digits, 0)
        then_val = _node_to_string(args[2], digits, 0)
        else_val = _node_to_string(args[3], digits, 0)
        return "if($cond ≥ 0, $then_val, $else_val)"
    end
    
    if func in INFIX_OPERATORS && length(args) == 2
        # Binary infix operator
        my_prec = get(OPERATOR_PRECEDENCE, func, 0)
        left = _node_to_string(args[1], digits, my_prec)
        right = _node_to_string(args[2], digits, my_prec)
        
        # Clean up the operator symbol for display
        op_str = string(func)
        if startswith(op_str, ".")
            op_str = " " * op_str * " "
        else
            op_str = " " * op_str * " "
        end
        
        expr = left * op_str * right
        
        # Add parentheses if needed
        if my_prec < parent_prec
            return "(" * expr * ")"
        else
            return "(" * expr * ")"  # Always parenthesize for clarity
        end
    else
        # Prefix notation: func(args...)
        args_str = join([_node_to_string(a, digits, 0) for a in args], ", ")
        return string(func) * "(" * args_str * ")"
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# LaTeX Representation
# ───────────────────────────────────────────────────────────────────────────────

const LATEX_FUNCTION_MAP = Dict(
    # Greek letters and special names
    :alpha => "\\alpha",
    :beta => "\\beta",
    :gamma => "\\gamma",
    :delta => "\\delta",
    :epsilon => "\\epsilon",
    :theta => "\\theta",
    :lambda => "\\lambda",
    :mu => "\\mu",
    :sigma => "\\sigma",
    :pi => "\\pi",
    
    # Common functions
    :sin => "\\sin",
    :cos => "\\cos",
    :tan => "\\tan",
    :exp => "\\exp",
    :log => "\\log",
    :ln => "\\ln",
    :sqrt => "\\sqrt",
    :abs => "\\left|",  # Special handling needed
    
    # Statistical functions
    :mean => "\\mu",
    :mean_v => "\\mu",
    :sum => "\\Sigma",
    :sum_v => "\\Sigma",
    :std => "\\sigma",
    :std_v => "\\sigma",
    :var => "\\text{Var}",
    :var_v => "\\text{Var}",
    :max => "\\max",
    :max_v => "\\max",
    :min => "\\min",
    :min_v => "\\min",
)

"""
    node_to_latex(node::AbstractNode; digits::Int=3) -> String

Convert an expression tree to LaTeX math notation.

# Examples
```julia
tree = FunctionNode(:/, 
    FunctionNode(:+, Variable(:x), Constant(1.0)),
    FunctionNode(:sqrt, Variable(:y))
)
node_to_latex(tree)  # "\\frac{x + 1}{\\sqrt{y}}"
```
"""
function node_to_latex(node::AbstractNode; digits::Int=3)
    return _node_to_latex(node, digits)
end

function _node_to_latex(node::Constant, digits::Int)
    val = node.value
    if val == round(val) && abs(val) < 1e6
        return string(Int(val))
    else
        return string(round(val; digits=digits))
    end
end

function _node_to_latex(node::Variable, digits::Int)
    name = string(node.name)
    # Check if it's a Greek letter
    sym = node.name
    if haskey(LATEX_FUNCTION_MAP, sym)
        return LATEX_FUNCTION_MAP[sym]
    elseif length(name) > 1
        # Multi-character variable: use \text{} or subscript
        if occursin("_", name)
            parts = split(name, "_"; limit=2)
            return parts[1] * "_{" * parts[2] * "}"
        else
            return "\\text{" * name * "}"
        end
    else
        return name
    end
end

function _node_to_latex(node::FunctionNode, digits::Int)
    func = node.func
    args = node.children
    
    # Handle ifelse specially: use piecewise notation
    if func == :ifelse && length(args) == 3
        cond = _node_to_latex(args[1], digits)
        then_val = _node_to_latex(args[2], digits)
        else_val = _node_to_latex(args[3], digits)
        return "\\begin{cases} " * then_val * " & \\text{if } " * cond * " \\geq 0 \\\\ " * else_val * " & \\text{otherwise} \\end{cases}"
    end
    
    # Handle division specially: use \frac
    if func == :/ && length(args) == 2
        num = _node_to_latex(args[1], digits)
        den = _node_to_latex(args[2], digits)
        return "\\frac{" * num * "}{" * den * "}"
    end
    
    # Handle power specially
    if (func == :^ || func == :.^) && length(args) == 2
        base = _node_to_latex(args[1], digits)
        exp = _node_to_latex(args[2], digits)
        # Wrap base in parens if it's complex
        if args[1] isa FunctionNode
            base = "\\left(" * base * "\\right)"
        end
        return base * "^{" * exp * "}"
    end
    
    # Handle sqrt
    if func == :sqrt && length(args) == 1
        arg = _node_to_latex(args[1], digits)
        return "\\sqrt{" * arg * "}"
    end
    
    # Handle absolute value
    if func == :abs && length(args) == 1
        arg = _node_to_latex(args[1], digits)
        return "\\left|" * arg * "\\right|"
    end
    
    # Handle infix operators
    if func in INFIX_OPERATORS && length(args) == 2
        left = _node_to_latex(args[1], digits)
        right = _node_to_latex(args[2], digits)
        
        op_str = if func == :* || func == :.*
            " \\cdot "
        elseif func == :+ || func == :.+
            " + "
        elseif func == :- || func == :.-
            " - "
        else
            " " * string(func) * " "
        end
        
        return "\\left(" * left * op_str * right * "\\right)"
    end
    
    # Element-wise operations: add dot notation
    func_str = string(func)
    if startswith(func_str, "ew_")
        # Element-wise function from grammar
        base_func = Symbol(func_str[4:end])
        func_str = string(base_func) * "_{\\cdot}"
    end
    
    # Look up LaTeX name
    latex_func = get(LATEX_FUNCTION_MAP, func, "\\text{" * func_str * "}")
    
    # Format arguments
    args_latex = [_node_to_latex(a, digits) for a in args]
    
    if length(args) == 1
        return latex_func * "\\left(" * args_latex[1] * "\\right)"
    else
        return latex_func * "\\left(" * join(args_latex, ", ") * "\\right)"
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Show methods for REPL display
# ───────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, node::Constant)
    print(io, "Constant(", node.value, ")")
end

function Base.show(io::IO, node::Variable)
    if istyped(node)
        print(io, "Variable(:", node.name, ", :", vartype(node), ")")
    else
        print(io, "Variable(:", node.name, ")")
    end
end

function Base.show(io::IO, node::FunctionNode)
    print(io, node_to_string(node))
end

function Base.show(io::IO, ::MIME"text/plain", node::AbstractNode)
    println(io, "Expression tree ($(count_nodes(node)) nodes, depth $(tree_depth(node))):")
    print(io, "  ", node_to_string(node))
end

# ───────────────────────────────────────────────────────────────────────────────
# Tree visualization (ASCII art)
# ───────────────────────────────────────────────────────────────────────────────

"""
    print_tree(node::AbstractNode; io::IO=stdout, indent::Int=0)

Print a tree in a hierarchical format showing the structure.

# Examples
```julia
tree = FunctionNode(:+, Variable(:x), FunctionNode(:*, Constant(2.0), Variable(:y)))
print_tree(tree)
# Output:
# +
#   x
#   *
#     2.0
#     y
```
"""
function print_tree(node::AbstractNode; io::IO=stdout, indent::Int=0)
    prefix = "  " ^ indent
    
    if node isa Constant
        println(io, prefix, node.value)
    elseif node isa Variable
        println(io, prefix, node.name)
    else
        println(io, prefix, node.func)
        for child in node.children
            print_tree(child; io=io, indent=indent+1)
        end
    end
end

"""
    tree_to_string_block(node::AbstractNode) -> String

Return the tree visualization as a string.
"""
function tree_to_string_block(node::AbstractNode)
    io = IOBuffer()
    print_tree(node; io=io)
    return String(take!(io))
end
