# Core Types & Trees

## Expression Nodes

All expression trees are composed of [`AbstractNode`](@ref) subtypes:

- **[`Constant`](@ref)`(value::Float64)`** — literal numeric values
- **[`Variable`](@ref)`(name::Symbol)`** — variable references (untyped)
- **[`Variable`](@ref)`(name::Symbol, type)`** — variable references (typed)
- **[`FunctionNode`](@ref)`(func::Symbol, children...)`** — function applications

### Building Trees Manually

```julia
x = Variable(:x)
tree = FunctionNode(:+,
    FunctionNode(:sin, x),
    Constant(1.0)
)
```

## Type Queries

```julia
isconstant(node)    # true for Constant
isvariable(node)    # true for Variable
isfunction(node)    # true for FunctionNode
isterminal(node)    # true for Constant or Variable
vartype(node)       # type of a Variable
istyped(node)       # true if Variable has a type
children(node)      # child nodes of a FunctionNode
arity(node)         # number of children
```

## Tree Utilities

### Size and Structure

```julia
count_nodes(node)        # Total node count
tree_depth(node)         # Maximum depth
tree_size_stats(node)    # Comprehensive statistics
```

### Traversal

```julia
flatten_tree(node)       # All nodes in pre-order
indexed_nodes(node)      # (index, node) pairs
terminals(node)          # Leaf nodes only
nonterminals(node)       # Internal nodes only
```

### Collection

```julia
collect_variables(node)  # Set of variable names
collect_constants(node)  # Vector of constant values
collect_functions(node)  # Vector of function symbols
```

### Manipulation

```julia
copy_tree(node)                      # Deep copy
replace_subtree(tree, old, new)      # Substitute subtrees
map_tree(f, node)                    # Transform all nodes
get_subtree_at_index(node, i)        # Access by index
random_subtree(node)                 # Random selection
random_subtree_index(node)           # Random index
```

### Printing

```julia
node_to_string(node)    # Human-readable: "(sin(x) + 1)"
node_to_latex(node)     # LaTeX: "\sin(x) + 1"
print_tree(node)        # Hierarchical ASCII view
tree_to_string_block(node)  # Block format
```

## Types Reference

```@docs
AbstractNode
Constant
Variable
FunctionNode
isconstant
isvariable
isfunction
isterminal
vartype
istyped
children
arity
copy_tree
count_nodes
tree_depth
tree_size_stats
flatten_tree
indexed_nodes
terminals
nonterminals
collect_variables
collect_constants
collect_functions
replace_subtree
map_tree
get_subtree_at_index
random_subtree
random_subtree_index
node_to_string
node_to_latex
print_tree
tree_to_string_block
```
