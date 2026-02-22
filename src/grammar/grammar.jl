# ═══════════════════════════════════════════════════════════════════════════════
# Grammar: Core Types and Construction
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
# Operator Specification
# ───────────────────────────────────────────────────────────────────────────────

"""
    OperatorSpec

Complete specification of an operator in the grammar.
"""
struct OperatorSpec
    name::Symbol
    func::Function
    arity::Int
    input_types::Vector{Symbol}
    output_type::Symbol
    safe_func::Function
    complexity::Float64
end

function Base.show(io::IO, op::OperatorSpec)
    if isempty(op.input_types) || all(t -> t == :Any, op.input_types)
        print(io, "$(op.name) (arity=$(op.arity))")
    else
        types_str = join(op.input_types, ", ")
        print(io, "$(op.name)($types_str) → $(op.output_type)")
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Variable Specification
# ───────────────────────────────────────────────────────────────────────────────

"""
    VariableSpec

Specification for a variable in the grammar.
"""
struct VariableSpec
    name::Symbol
    type::Symbol
    complexity::Float64
end

VariableSpec(name::Symbol) = VariableSpec(name, :Any, 1.0)
VariableSpec(name::Symbol, type::Symbol) = VariableSpec(name, type, 1.0)

function Base.show(io::IO, v::VariableSpec)
    if v.type == :Any
        print(io, ":$(v.name)")
    else
        print(io, ":$(v.name) :: $(v.type)")
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Constant Specification
# ───────────────────────────────────────────────────────────────────────────────

"""
    ConstantSpec

Specification for how to sample constants.
"""
struct ConstantSpec
    type::Symbol
    sampler::Function
    complexity::Float64
end

function ConstantSpec(range::Tuple{<:Real, <:Real}; type::Symbol=:Any, complexity::Float64=1.0)
    lo, hi = Float64.(range)
    ConstantSpec(type, () -> lo + rand() * (hi - lo), complexity)
end

# ───────────────────────────────────────────────────────────────────────────────
# Grammar Struct
# ───────────────────────────────────────────────────────────────────────────────

"""
    Grammar

A complete grammar specification for symbolic optimization.

# Two Modes

## Simple Mode (like SymbolicRegression.jl)
```julia
g = Grammar(
    binary_operators = [+, -, *, /],
    unary_operators = [sin, cos, exp],
    variables = [:x, :y],
    constant_range = (-2.0, 2.0),
)
```

## Typed Mode (for complex domains)
```julia
g = Grammar(
    types = [:Scalar, :Vector],
    variables = [:ps => :Vector, :n => :Scalar],
    operators = [
        (mean, [:Vector] => :Scalar),
        (+, [:Scalar, :Scalar] => :Scalar),
        (.^, [:Vector, :Scalar] => :Vector),
    ],
    constant_types = [:Scalar],
    output_type = :Scalar,
)
```
"""
struct Grammar
    # Core data
    operators::Vector{OperatorSpec}
    variables::Vector{VariableSpec}
    constants::Vector{ConstantSpec}
    types::Set{Symbol}
    output_types::Set{Symbol}
    is_typed::Bool
    constant_prob::Float64  # Probability of generating a constant vs variable (0.0-1.0)
    
    # Cached lookups
    _op_by_name::Dict{Symbol, Vector{OperatorSpec}}
    _op_by_arity::Dict{Int, Vector{OperatorSpec}}
    _op_by_output::Dict{Symbol, Vector{OperatorSpec}}
    _var_by_type::Dict{Symbol, Vector{VariableSpec}}
    _const_by_type::Dict{Symbol, Vector{ConstantSpec}}
end

# ───────────────────────────────────────────────────────────────────────────────
# Grammar Constructor
# ───────────────────────────────────────────────────────────────────────────────

"""
    Grammar(; kwargs...)

Construct a Grammar specification.

# Simple Mode Keywords
- `binary_operators`: Vector of binary functions (e.g., `[+, -, *, /]`)
- `unary_operators`: Vector of unary functions (e.g., `[sin, cos, exp]`)
- `ternary_operators`: Vector of ternary functions (e.g., `[ifelse]` for conditionals)
- `variables`: Vector of variable names as Symbols
- `constant_range`: Tuple `(lo, hi)` for sampling constants

# Typed Mode Keywords
- `types`: Vector of type symbols triggers typed mode
- `operators`: Vector of `(func, signature)` pairs
- `variables`: Vector of `name => type` pairs
- `constant_types`: Which types can have constants
- `output_type` or `output_types`: Valid output types

# Terminal Selection Keywords (both modes)
- `constant_prob`: Probability of generating a constant vs variable when creating
  a terminal node (default: 0.3). Set to 0.0 for variables only, 1.0 for constants only.

# Complexity Keywords (both modes)
- `complexity_of_operators`: Dict mapping operators to complexity, or single value
- `complexity_of_variables`: Complexity for variables (default: 1.0)
- `complexity_of_constants`: Complexity for constants (default: 1.0)

# Example with Conditionals (for piecewise functions)
```julia
# Enable discovery of piecewise functions like Crupi's z measure
g = Grammar(
    binary_operators = [+, -, *, /],
    unary_operators = [safe_step, safe_abs],  # step function enables conditionals
    ternary_operators = [:ifelse],            # ifelse(cond, then, else)
    variables = [:pH, :pE, :pH_E, ...],
    constant_prob = 0.0,  # No arbitrary constants
)
```
"""
function Grammar(;
    # Simple mode
    binary_operators::Union{Vector, Nothing} = nothing,
    unary_operators::Union{Vector, Nothing} = nothing,
    ternary_operators::Union{Vector, Nothing} = nothing,
    
    # Typed mode
    types::Union{Vector{Symbol}, Nothing} = nothing,
    operators::Union{Vector, Nothing} = nothing,
    
    # Both modes
    variables::Union{Vector, Nothing} = nothing,
    constant_range::Union{Tuple{<:Real, <:Real}, Nothing} = (-2.0, 2.0),
    constant_types::Union{Vector{Symbol}, Nothing} = nothing,
    output_type::Union{Symbol, Nothing} = nothing,
    output_types::Union{Vector{Symbol}, Nothing} = nothing,
    
    # Terminal selection
    constant_prob::Real = 0.3,  # 30% constants, 70% variables (matches original paper)
    
    # Complexity
    complexity_of_operators::Union{Dict, Real, Nothing} = nothing,
    complexity_of_variables::Real = 1.0,
    complexity_of_constants::Real = 1.0,
)
    # Validate constant_prob
    if !(0.0 <= constant_prob <= 1.0)
        error("constant_prob must be between 0.0 and 1.0, got: $constant_prob")
    end
    
    # Determine mode
    is_typed = types !== nothing
    
    if is_typed
        return _build_typed_grammar(
            types, operators, variables, constant_range, constant_types,
            output_type, output_types, complexity_of_operators,
            complexity_of_variables, complexity_of_constants, Float64(constant_prob)
        )
    else
        return _build_simple_grammar(
            binary_operators, unary_operators, ternary_operators, variables, constant_range,
            complexity_of_operators, complexity_of_variables, complexity_of_constants,
            Float64(constant_prob)
        )
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Simple Grammar Builder
# ───────────────────────────────────────────────────────────────────────────────

function _build_simple_grammar(
    binary_operators, unary_operators, ternary_operators, variables, constant_range,
    complexity_of_ops, complexity_of_vars, complexity_of_consts, constant_prob
)
    # Defaults
    binary_ops = something(binary_operators, Function[+, -, *, /])
    unary_ops = something(unary_operators, Function[])
    ternary_ops = something(ternary_operators, [])
    vars = something(variables, Symbol[:x])
    
    # Build operator specs
    op_specs = OperatorSpec[]
    
    for func in binary_ops
        spec = _create_operator_spec(func, 2, Symbol[], :Any, complexity_of_ops)
        push!(op_specs, spec)
    end
    
    for func in unary_ops
        spec = _create_operator_spec(func, 1, Symbol[], :Any, complexity_of_ops)
        push!(op_specs, spec)
    end
    
    for func in ternary_ops
        spec = _create_operator_spec(func, 3, Symbol[], :Any, complexity_of_ops)
        push!(op_specs, spec)
    end
    
    # Build variable specs
    var_specs = [VariableSpec(v, :Any, Float64(complexity_of_vars)) for v in vars]
    
    # Build constant specs
    const_specs = ConstantSpec[]
    if constant_range !== nothing
        push!(const_specs, ConstantSpec(constant_range; type=:Any, complexity=Float64(complexity_of_consts)))
    end
    
    # Build grammar
    return _finalize_grammar(op_specs, var_specs, const_specs, Set([:Any]), Set([:Any]), false, constant_prob)
end

# ───────────────────────────────────────────────────────────────────────────────
# Typed Grammar Builder
# ───────────────────────────────────────────────────────────────────────────────

function _build_typed_grammar(
    types, operators, variables, constant_range, constant_types,
    output_type, output_types, complexity_of_ops,
    complexity_of_vars, complexity_of_consts, constant_prob
)
    type_set = Set(types)
    
    # Parse operators
    op_specs = OperatorSpec[]
    if operators !== nothing
        for op in operators
            spec = _parse_typed_operator(op, type_set, complexity_of_ops)
            push!(op_specs, spec)
        end
    end
    
    # Parse variables
    var_specs = VariableSpec[]
    if variables !== nothing
        for v in variables
            if v isa Symbol
                push!(var_specs, VariableSpec(v, :Any, Float64(complexity_of_vars)))
            elseif v isa Pair
                name, typ = v.first, v.second
                if typ ∉ type_set
                    error("Unknown type '$typ' for variable '$name'. Known types: $type_set")
                end
                push!(var_specs, VariableSpec(name, typ, Float64(complexity_of_vars)))
            else
                error("Variable must be Symbol or Pair, got: $(typeof(v))")
            end
        end
    end
    
    # Build constant specs
    const_specs = ConstantSpec[]
    if constant_range !== nothing
        ctypes = something(constant_types, [:Any])
        for t in ctypes
            push!(const_specs, ConstantSpec(constant_range; type=t, complexity=Float64(complexity_of_consts)))
        end
    end
    
    # Output types
    out_types = if output_types !== nothing
        Set(output_types)
    elseif output_type !== nothing
        Set([output_type])
    else
        type_set
    end
    
    return _finalize_grammar(op_specs, var_specs, const_specs, type_set, out_types, true, constant_prob)
end

function _parse_typed_operator(op, type_set::Set{Symbol}, complexity_of_ops)
    # Handle: (func, [:In1, :In2] => :Out) or (func, [:In1, :In2] => :Out, impl)
    if !(op isa Tuple) || length(op) < 2
        error("Typed operator must be (func, signature) or (func, signature, impl). Got: $op")
    end
    
    func = op[1]
    sig = op[2]
    impl = length(op) >= 3 ? op[3] : nothing
    
    if !(sig isa Pair)
        error("Signature must be InputTypes => OutputType. Got: $sig")
    end
    
    input_types = sig.first isa Vector ? sig.first : [sig.first]
    output_type = sig.second
    
    # Validate types
    for t in input_types
        if t ∉ type_set && t != :Any
            error("Unknown input type '$t'. Known types: $type_set")
        end
    end
    if output_type ∉ type_set && output_type != :Any
        error("Unknown output type '$output_type'. Known types: $type_set")
    end
    
    arity = length(input_types)
    return _create_operator_spec(func, arity, input_types, output_type, complexity_of_ops; impl=impl)
end

# ───────────────────────────────────────────────────────────────────────────────
# Operator Spec Creation
# ───────────────────────────────────────────────────────────────────────────────

function _create_operator_spec(func, arity::Int, input_types::Vector, output_type::Symbol,
                               complexity_of_ops; impl=nothing)
    # Get name
    name = _get_operator_name(func)
    
    # Determine actual function to use
    actual_func = if impl !== nothing
        impl
    elseif func isa Function
        func
    elseif func isa Symbol
        # Try to find in registry
        get(SAFE_IMPLEMENTATIONS, func, nothing)
    else
        nothing
    end
    
    if actual_func === nothing
        error("Cannot resolve function for operator: $func")
    end
    
    # Get safe version
    safe_func = _get_safe_version(actual_func, arity)
    
    # Complexity
    complexity = if complexity_of_ops isa Dict
        get(complexity_of_ops, func, get(complexity_of_ops, name, 1.0))
    elseif complexity_of_ops isa Real
        Float64(complexity_of_ops)
    else
        1.0
    end
    
    return OperatorSpec(name, actual_func, arity, Symbol[input_types...], output_type, safe_func, Float64(complexity))
end

function _get_operator_name(func)::Symbol
    if func isa Symbol
        return func
    elseif func isa Function
        return nameof(func)
    else
        return Symbol(string(func))
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Grammar Finalization
# ───────────────────────────────────────────────────────────────────────────────

function _finalize_grammar(op_specs, var_specs, const_specs, type_set, out_types, is_typed, constant_prob)
    # Build caches
    op_by_name = Dict{Symbol, Vector{OperatorSpec}}()
    op_by_arity = Dict{Int, Vector{OperatorSpec}}()
    op_by_output = Dict{Symbol, Vector{OperatorSpec}}()
    
    for op in op_specs
        # By name
        if !haskey(op_by_name, op.name)
            op_by_name[op.name] = OperatorSpec[]
        end
        push!(op_by_name[op.name], op)
        
        # By arity
        if !haskey(op_by_arity, op.arity)
            op_by_arity[op.arity] = OperatorSpec[]
        end
        push!(op_by_arity[op.arity], op)
        
        # By output type
        if !haskey(op_by_output, op.output_type)
            op_by_output[op.output_type] = OperatorSpec[]
        end
        push!(op_by_output[op.output_type], op)
    end
    
    var_by_type = Dict{Symbol, Vector{VariableSpec}}()
    for v in var_specs
        if !haskey(var_by_type, v.type)
            var_by_type[v.type] = VariableSpec[]
        end
        push!(var_by_type[v.type], v)
    end
    
    const_by_type = Dict{Symbol, Vector{ConstantSpec}}()
    for c in const_specs
        if !haskey(const_by_type, c.type)
            const_by_type[c.type] = ConstantSpec[]
        end
        push!(const_by_type[c.type], c)
    end
    
    return Grammar(
        op_specs, var_specs, const_specs, type_set, out_types, is_typed, constant_prob,
        op_by_name, op_by_arity, op_by_output, var_by_type, const_by_type
    )
end

# ───────────────────────────────────────────────────────────────────────────────
# Grammar Accessors
# ───────────────────────────────────────────────────────────────────────────────

"""Check if grammar uses types."""
is_typed(g::Grammar) = g.is_typed

"""Get all operator specs."""
all_operators(g::Grammar) = g.operators

"""Get all variable specs."""
all_variables(g::Grammar) = g.variables

"""Get all constant specs."""
all_constants(g::Grammar) = g.constants

"""Get operators by name."""
operators_by_name(g::Grammar, name::Symbol) = get(g._op_by_name, name, OperatorSpec[])

"""Get operators by arity."""
operators_by_arity(g::Grammar, arity::Int) = get(g._op_by_arity, arity, OperatorSpec[])

"""Get unary operators."""
unary_operators(g::Grammar) = operators_by_arity(g, 1)

"""Get binary operators."""
binary_operators(g::Grammar) = operators_by_arity(g, 2)

"""Get ternary operators (for conditionals like ifelse)."""
ternary_operators(g::Grammar) = operators_by_arity(g, 3)

"""Get operators that produce a given type."""
operators_producing(g::Grammar, type::Symbol) = get(g._op_by_output, type, OperatorSpec[])

"""Get variables of a given type."""
variables_of_type(g::Grammar, type::Symbol) = get(g._var_by_type, type, VariableSpec[])

"""Get constant specs for a given type."""
constants_of_type(g::Grammar, type::Symbol) = get(g._const_by_type, type, ConstantSpec[])

"""Check if a type exists in the grammar."""
has_type(g::Grammar, type::Symbol) = type ∈ g.types || type == :Any

"""Check if an operator exists."""
has_operator(g::Grammar, name::Symbol) = haskey(g._op_by_name, name)

"""Get the number of operators."""
num_operators(g::Grammar) = length(g.operators)

"""Get the number of variables."""
num_variables(g::Grammar) = length(g.variables)

"""Sample a constant value."""
function sample_constant(g::Grammar, type::Symbol=:Any)
    specs = if type == :Any
        g.constants
    else
        cs = constants_of_type(g, type)
        isempty(cs) ? g.constants : cs
    end
    
    if isempty(specs)
        return 0.0
    end
    
    spec = rand(specs)
    return spec.sampler()
end

# ───────────────────────────────────────────────────────────────────────────────
# Grammar Display
# ───────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, g::Grammar)
    mode = g.is_typed ? "Typed" : "Simple"
    print(io, "Grammar($mode, $(num_operators(g)) ops, $(num_variables(g)) vars)")
end

function Base.show(io::IO, ::MIME"text/plain", g::Grammar)
    mode = g.is_typed ? "Typed" : "Simple"
    println(io, "$mode Grammar")
    println(io, "─"^50)
    
    if g.is_typed && !isempty(g.types)
        println(io, "Types: ", join(sort(collect(g.types)), ", "))
        println(io)
    end
    
    # Group operators by arity
    println(io, "Operators ($(num_operators(g)) total):")
    for arity in sort(collect(keys(g._op_by_arity)))
        ops = g._op_by_arity[arity]
        label = arity == 1 ? "Unary" : arity == 2 ? "Binary" : "$arity-ary"
        println(io, "  $label:")
        for op in ops
            if g.is_typed && !isempty(op.input_types) && !all(t -> t == :Any, op.input_types)
                types_str = join(op.input_types, ", ")
                println(io, "    $(op.name)($types_str) → $(op.output_type)")
            else
                println(io, "    $(op.name)")
            end
        end
    end
    
    println(io)
    println(io, "Variables ($(num_variables(g)) total):")
    for v in g.variables
        if g.is_typed && v.type != :Any
            println(io, "  $(v.name) :: $(v.type)")
        else
            println(io, "  $(v.name)")
        end
    end
    
    if !isempty(g.constants)
        println(io)
        println(io, "Constants: $(length(g.constants)) sampler(s)")
    end
    
    if g.is_typed
        println(io)
        println(io, "Output types: ", join(sort(collect(g.output_types)), ", "))
    end
end
