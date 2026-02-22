# ═══════════════════════════════════════════════════════════════════════════════
# Core Types for Symbolic Expression Trees
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AbstractNode

Abstract base type for all nodes in symbolic expression trees.

Concrete subtypes:
- [`Constant`](@ref): Literal numeric values
- [`Variable`](@ref): Named variable references
- [`FunctionNode`](@ref): Function applications with children
"""
abstract type AbstractNode end

# ───────────────────────────────────────────────────────────────────────────────
# Constant Node
# ───────────────────────────────────────────────────────────────────────────────

"""
    Constant <: AbstractNode
    Constant(value::Float64)

A constant (literal) numeric value in the expression tree.

# Examples
```julia
c = Constant(3.14)
c.value  # 3.14
```
"""
struct Constant <: AbstractNode
    value::Float64
end

# Allow construction from any Real
Constant(x::Real) = Constant(Float64(x))

# ───────────────────────────────────────────────────────────────────────────────
# Variable Node
# ───────────────────────────────────────────────────────────────────────────────

"""
    Variable{T} <: AbstractNode
    Variable(name::Symbol, type::T)
    Variable(name::Symbol)  # untyped

A variable reference in the expression tree.

The type parameter `T` encodes the variable's type in typed grammars
(e.g., `Symbol` for `:Scalar`, `:Vector`), or is `Nothing` for untyped grammars.

# Examples
```julia
# Untyped variable
x = Variable(:x)

# Typed variable
ps = Variable(:ps, :Vector)
vartype(ps)  # :Vector
```
"""
struct Variable{T} <: AbstractNode
    name::Symbol
    type::T
    
    Variable(name::Symbol, type::T) where T = new{T}(name, type)
    Variable(name::Symbol) = new{Nothing}(name, nothing)
end

"""
    vartype(v::Variable) -> T

Return the type annotation of a variable, or `nothing` for untyped variables.
"""
vartype(v::Variable) = v.type

"""
    istyped(v::Variable) -> Bool

Return `true` if the variable has a type annotation.
"""
istyped(::Variable{Nothing}) = false
istyped(::Variable) = true

# ───────────────────────────────────────────────────────────────────────────────
# Function Node
# ───────────────────────────────────────────────────────────────────────────────

"""
    FunctionNode <: AbstractNode
    FunctionNode(func::Symbol, children::Vector{AbstractNode})
    FunctionNode(func::Symbol, children::AbstractNode...)

A function application node with child nodes as arguments.

# Examples
```julia
# Addition: x + y
add_node = FunctionNode(:+, Variable(:x), Variable(:y))

# Nested: sin(x + 1)
nested = FunctionNode(:sin, FunctionNode(:+, Variable(:x), Constant(1.0)))
```
"""
struct FunctionNode <: AbstractNode
    func::Symbol
    children::Vector{AbstractNode}
    
    function FunctionNode(func::Symbol, children::AbstractVector)
        new(func, convert(Vector{AbstractNode}, children))
    end
end

# Convenience constructor with varargs
FunctionNode(func::Symbol, children::AbstractNode...) = FunctionNode(func, collect(children))

# ───────────────────────────────────────────────────────────────────────────────
# Type Predicates
# ───────────────────────────────────────────────────────────────────────────────

"""
    isconstant(node::AbstractNode) -> Bool

Return `true` if the node is a `Constant`.
"""
isconstant(node::AbstractNode) = node isa Constant

"""
    isvariable(node::AbstractNode) -> Bool

Return `true` if the node is a `Variable`.
"""
isvariable(node::AbstractNode) = node isa Variable

"""
    isfunction(node::AbstractNode) -> Bool

Return `true` if the node is a `FunctionNode`.
"""
isfunction(node::AbstractNode) = node isa FunctionNode

"""
    isterminal(node::AbstractNode) -> Bool

Return `true` if the node is a terminal (Constant or Variable).
"""
isterminal(node::AbstractNode) = isconstant(node) || isvariable(node)

# ───────────────────────────────────────────────────────────────────────────────
# Accessors
# ───────────────────────────────────────────────────────────────────────────────

"""
    children(node::AbstractNode) -> Vector{AbstractNode}

Return the children of a node. Empty for terminals.
"""
children(node::FunctionNode) = node.children
children(::Constant) = AbstractNode[]
children(::Variable) = AbstractNode[]

"""
    arity(node::AbstractNode) -> Int

Return the number of children (arity) of a node.
"""
arity(node::FunctionNode) = length(node.children)
arity(::Constant) = 0
arity(::Variable) = 0

# ───────────────────────────────────────────────────────────────────────────────
# Equality and Hashing (for use in Sets/Dicts)
# ───────────────────────────────────────────────────────────────────────────────

Base.:(==)(a::Constant, b::Constant) = a.value == b.value
Base.:(==)(a::Variable, b::Variable) = a.name == b.name && a.type == b.type
Base.:(==)(a::FunctionNode, b::FunctionNode) = a.func == b.func && a.children == b.children
Base.:(==)(::AbstractNode, ::AbstractNode) = false

Base.hash(c::Constant, h::UInt) = hash(c.value, hash(:Constant, h))
Base.hash(v::Variable, h::UInt) = hash(v.type, hash(v.name, hash(:Variable, h)))
Base.hash(f::FunctionNode, h::UInt) = hash(f.children, hash(f.func, hash(:FunctionNode, h)))
