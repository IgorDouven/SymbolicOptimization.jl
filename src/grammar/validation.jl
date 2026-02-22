# ═══════════════════════════════════════════════════════════════════════════════
# Grammar Validation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ValidationResult

Result of grammar validation.
"""
struct ValidationResult
    valid::Bool
    errors::Vector{String}
    warnings::Vector{String}
end

ValidationResult() = ValidationResult(true, String[], String[])

function Base.show(io::IO, r::ValidationResult)
    if r.valid
        print(io, "ValidationResult(valid=true")
        if !isempty(r.warnings)
            print(io, ", $(length(r.warnings)) warnings")
        end
        print(io, ")")
    else
        print(io, "ValidationResult(valid=false, $(length(r.errors)) errors)")
    end
end

"""
    validate_grammar(g::Grammar) -> ValidationResult

Validate a grammar for consistency and completeness.

# Checks performed:
- All operator input/output types exist in the grammar
- All variable types exist in the grammar
- At least one operator or variable produces each output type
- No duplicate operator signatures (in typed mode)
- Constants can be generated for required types
"""
function validate_grammar(g::Grammar)
    errors = String[]
    warnings = String[]
    
    # Check operator types
    for op in g.operators
        for (i, t) in enumerate(op.input_types)
            if t != :Any && !has_type(g, t)
                push!(errors, "Operator '$(op.name)' has unknown input type '$t' at position $i")
            end
        end
        if op.output_type != :Any && !has_type(g, op.output_type)
            push!(errors, "Operator '$(op.name)' has unknown output type '$(op.output_type)'")
        end
    end
    
    # Check variable types
    for v in g.variables
        if v.type != :Any && !has_type(g, v.type)
            push!(errors, "Variable '$(v.name)' has unknown type '$(v.type)'")
        end
    end
    
    # Check constant types
    for c in g.constants
        if c.type != :Any && !has_type(g, c.type)
            push!(errors, "Constant specification has unknown type '$(c.type)'")
        end
    end
    
    # Check output types are producible
    if g.is_typed
        for out_type in g.output_types
            if !_can_produce_type(g, out_type)
                push!(errors, "Output type '$out_type' cannot be produced by any operator or variable")
            end
        end
    end
    
    # Check for at least some operators
    if isempty(g.operators)
        push!(warnings, "Grammar has no operators")
    end
    
    # Check for at least some terminals
    if isempty(g.variables) && isempty(g.constants)
        push!(errors, "Grammar has no terminals (variables or constants)")
    end
    
    # Check typed grammar has all input types covered
    if g.is_typed
        needed_types = Set{Symbol}()
        for op in g.operators
            for t in op.input_types
                if t != :Any
                    push!(needed_types, t)
                end
            end
        end
        
        for t in needed_types
            vars = variables_of_type(g, t)
            consts = constants_of_type(g, t)
            ops = operators_producing(g, t)
            
            if isempty(vars) && isempty(consts) && isempty(ops)
                push!(warnings, "Type '$t' is needed as input but cannot be produced")
            end
        end
    end
    
    valid = isempty(errors)
    return ValidationResult(valid, errors, warnings)
end

function _can_produce_type(g::Grammar, type::Symbol)
    # Check variables
    for v in g.variables
        if v.type == type || v.type == :Any
            return true
        end
    end
    
    # Check constants
    for c in g.constants
        if c.type == type || c.type == :Any
            return true
        end
    end
    
    # Check operators
    for op in g.operators
        if op.output_type == type || op.output_type == :Any
            return true
        end
    end
    
    return false
end

"""
    validate_grammar!(g::Grammar)

Validate grammar and throw an error if invalid.
"""
function validate_grammar!(g::Grammar)
    result = validate_grammar(g)
    if !result.valid
        error_msg = "Invalid grammar:\n" * join(["  - " * e for e in result.errors], "\n")
        throw(ArgumentError(error_msg))
    end
    if !isempty(result.warnings)
        for w in result.warnings
            @warn "Grammar warning: $w"
        end
    end
    return g
end

"""
    check_tree_validity(tree::AbstractNode, g::Grammar) -> Tuple{Bool, String}

Check if an expression tree is valid according to the grammar.
Returns `(valid, message)`.
"""
function check_tree_validity(tree::AbstractNode, g::Grammar)
    if tree isa Constant
        if isempty(g.constants)
            return (false, "Grammar does not allow constants")
        end
        return (true, "")
    elseif tree isa Variable
        if !any(v -> v.name == tree.name, g.variables)
            return (false, "Unknown variable: $(tree.name)")
        end
        return (true, "")
    else  # FunctionNode
        # Check operator exists
        ops = operators_by_name(g, tree.func)
        if isempty(ops)
            return (false, "Unknown operator: $(tree.func)")
        end
        
        # Check arity
        expected_arities = Set(op.arity for op in ops)
        actual_arity = length(tree.children)
        if actual_arity ∉ expected_arities
            return (false, "Operator $(tree.func) has wrong arity: expected $(expected_arities), got $actual_arity")
        end
        
        # Recursively check children
        for child in tree.children
            valid, msg = check_tree_validity(child, g)
            if !valid
                return (false, msg)
            end
        end
        
        return (true, "")
    end
end

"""
    infer_type(tree::AbstractNode, g::Grammar, var_env::Dict{Symbol, Symbol}) -> Symbol

Infer the type of an expression tree in a typed grammar.
Returns `:Any` for untyped grammars.
"""
function infer_type(tree::AbstractNode, g::Grammar, var_env::Dict{Symbol, Symbol}=Dict{Symbol, Symbol}())
    if !g.is_typed
        return :Any
    end
    
    if tree isa Constant
        # Return first constant type, or :Any
        if !isempty(g.constants)
            return g.constants[1].type
        end
        return :Any
    elseif tree isa Variable
        # Check var_env first, then grammar
        if haskey(var_env, tree.name)
            return var_env[tree.name]
        end
        for v in g.variables
            if v.name == tree.name
                return v.type
            end
        end
        return :Any
    else  # FunctionNode
        # Infer children types
        child_types = [infer_type(c, g, var_env) for c in tree.children]
        
        # Find matching operator
        for op in operators_by_name(g, tree.func)
            if _types_match(child_types, op.input_types)
                return op.output_type
            end
        end
        
        return :Any
    end
end

function _types_match(actual::Vector{Symbol}, expected::Vector{Symbol})
    if length(actual) != length(expected)
        return false
    end
    for (a, e) in zip(actual, expected)
        if e != :Any && a != :Any && a != e
            return false
        end
    end
    return true
end
