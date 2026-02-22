# ═══════════════════════════════════════════════════════════════════════════════
# Tree Manipulation Utilities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    copy_tree(node::AbstractNode) -> AbstractNode

Create a deep copy of the expression tree.

# Examples
```julia
original = FunctionNode(:+, Variable(:x), Constant(1.0))
copied = copy_tree(original)
original == copied  # true
original === copied # false (different objects)
```
"""
function copy_tree(node::AbstractNode)
    if node isa Constant
        return Constant(node.value)
    elseif node isa Variable
        return Variable(node.name, vartype(node))
    else
        return FunctionNode(node.func, [copy_tree(c) for c in node.children])
    end
end

"""
    count_nodes(node::AbstractNode) -> Int

Count the total number of nodes in the tree.

# Examples
```julia
tree = FunctionNode(:+, Variable(:x), Constant(1.0))
count_nodes(tree)  # 3
```
"""
function count_nodes(node::AbstractNode)
    if isterminal(node)
        return 1
    else
        return 1 + sum(count_nodes(c) for c in node.children)
    end
end

"""
    tree_depth(node::AbstractNode) -> Int

Compute the maximum depth of the tree. Terminals have depth 1.

# Examples
```julia
tree = FunctionNode(:sin, FunctionNode(:+, Variable(:x), Constant(1.0)))
tree_depth(tree)  # 3
```
"""
function tree_depth(node::AbstractNode)
    if isterminal(node)
        return 1
    else
        child_depths = [tree_depth(c) for c in node.children]
        return 1 + (isempty(child_depths) ? 0 : maximum(child_depths))
    end
end

"""
    flatten_tree(node::AbstractNode) -> Vector{AbstractNode}

Return all nodes in the tree in pre-order traversal (root first, then children recursively).

# Examples
```julia
tree = FunctionNode(:+, Variable(:x), Constant(1.0))
nodes = flatten_tree(tree)  # [FunctionNode(:+, ...), Variable(:x), Constant(1.0)]
length(nodes)  # 3
```
"""
function flatten_tree(node::AbstractNode)
    nodes = AbstractNode[node]
    if node isa FunctionNode
        for child in node.children
            append!(nodes, flatten_tree(child))
        end
    end
    return nodes
end

"""
    collect_variables(node::AbstractNode) -> Set{Symbol}

Return the set of all variable names used in the tree.

# Examples
```julia
tree = FunctionNode(:+, Variable(:x), FunctionNode(:*, Variable(:x), Variable(:y)))
collect_variables(tree)  # Set([:x, :y])
```
"""
function collect_variables(node::AbstractNode)
    vars = Set{Symbol}()
    for n in flatten_tree(node)
        if n isa Variable
            push!(vars, n.name)
        end
    end
    return vars
end

"""
    collect_constants(node::AbstractNode) -> Vector{Float64}

Return all constant values in the tree (with duplicates, in pre-order).

# Examples
```julia
tree = FunctionNode(:+, Constant(1.0), FunctionNode(:*, Constant(2.0), Constant(1.0)))
collect_constants(tree)  # [1.0, 2.0, 1.0]
```
"""
function collect_constants(node::AbstractNode)
    return [n.value for n in flatten_tree(node) if n isa Constant]
end

"""
    collect_functions(node::AbstractNode) -> Vector{Symbol}

Return all function symbols used in the tree (with duplicates, in pre-order).

# Examples
```julia
tree = FunctionNode(:+, Variable(:x), FunctionNode(:+, Constant(1.0), Constant(2.0)))
collect_functions(tree)  # [:+, :+]
```
"""
function collect_functions(node::AbstractNode)
    return [n.func for n in flatten_tree(node) if n isa FunctionNode]
end

"""
    replace_subtree(tree, old_node, new_node) -> AbstractNode

Replace `old_node` (by identity, i.e., `===`) with `new_node` in the tree.
Returns a new tree; the original is not modified.

# Examples
```julia
x = Variable(:x)
tree = FunctionNode(:+, x, Constant(1.0))
new_tree = replace_subtree(tree, x, Constant(2.0))
# new_tree is now: (+ 2.0 1.0)
```
"""
function replace_subtree(tree::AbstractNode, old_node::AbstractNode, new_node::AbstractNode)
    if tree === old_node
        return copy_tree(new_node)
    elseif tree isa FunctionNode
        new_children = [replace_subtree(c, old_node, new_node) for c in tree.children]
        return FunctionNode(tree.func, new_children)
    else
        return tree  # Constant or Variable, unchanged
    end
end

"""
    map_tree(f, node::AbstractNode) -> AbstractNode

Apply function `f` to each node in the tree, building a new tree from the results.
The function `f` receives each node and should return an `AbstractNode`.

The mapping is applied bottom-up: children are mapped first, then the parent.

# Examples
```julia
# Double all constants
tree = FunctionNode(:+, Constant(1.0), Constant(2.0))
doubled = map_tree(tree) do node
    node isa Constant ? Constant(node.value * 2) : node
end
# doubled is now: (+ 2.0 4.0)
```
"""
function map_tree(f, node::AbstractNode)
    if node isa FunctionNode
        # First map children, then apply f to the reconstructed parent
        new_children = [map_tree(f, c) for c in node.children]
        new_node = FunctionNode(node.func, new_children)
        return f(new_node)
    else
        return f(node)
    end
end

"""
    get_subtree_at_index(node::AbstractNode, idx::Int) -> AbstractNode

Get the subtree at the given index (1-based, pre-order traversal).

# Examples
```julia
tree = FunctionNode(:+, Variable(:x), Constant(1.0))
get_subtree_at_index(tree, 1)  # The FunctionNode itself
get_subtree_at_index(tree, 2)  # Variable(:x)
get_subtree_at_index(tree, 3)  # Constant(1.0)
```
"""
function get_subtree_at_index(node::AbstractNode, idx::Int)
    nodes = flatten_tree(node)
    if idx < 1 || idx > length(nodes)
        throw(BoundsError(nodes, idx))
    end
    return nodes[idx]
end

"""
    indexed_nodes(node::AbstractNode) -> Vector{Tuple{Int, AbstractNode}}

Return (index, node) pairs for all nodes in pre-order traversal.

# Examples
```julia
tree = FunctionNode(:+, Variable(:x), Constant(1.0))
for (i, n) in indexed_nodes(tree)
    println("\$i: \$n")
end
```
"""
function indexed_nodes(node::AbstractNode)
    return collect(enumerate(flatten_tree(node)))
end

"""
    random_subtree(node::AbstractNode; rng=Random.GLOBAL_RNG) -> AbstractNode

Select a random subtree from the tree.

# Examples
```julia
tree = FunctionNode(:+, Variable(:x), Constant(1.0))
subtree = random_subtree(tree)  # Could be any of the 3 nodes
```
"""
function random_subtree(node::AbstractNode; rng=Random.GLOBAL_RNG)
    nodes = flatten_tree(node)
    return rand(rng, nodes)
end

"""
    random_subtree_index(node::AbstractNode; rng=Random.GLOBAL_RNG) -> Int

Select a random subtree index (1-based, pre-order).
"""
function random_subtree_index(node::AbstractNode; rng=Random.GLOBAL_RNG)
    n = count_nodes(node)
    return rand(rng, 1:n)
end

"""
    terminals(node::AbstractNode) -> Vector{AbstractNode}

Return all terminal nodes (Constants and Variables) in the tree.
"""
function terminals(node::AbstractNode)
    return filter(isterminal, flatten_tree(node))
end

"""
    nonterminals(node::AbstractNode) -> Vector{AbstractNode}

Return all non-terminal nodes (FunctionNodes) in the tree.
"""
function nonterminals(node::AbstractNode)
    return filter(isfunction, flatten_tree(node))
end

"""
    tree_size_stats(node::AbstractNode) -> NamedTuple

Return various size statistics about the tree.

# Returns
- `nodes`: Total node count
- `depth`: Maximum depth
- `terminals`: Number of terminal nodes
- `functions`: Number of function nodes
- `constants`: Number of constant nodes
- `variables`: Number of variable nodes
- `unique_vars`: Number of unique variable names
- `unique_funcs`: Number of unique function symbols
"""
function tree_size_stats(node::AbstractNode)
    all_nodes = flatten_tree(node)
    
    n_const = count(n -> n isa Constant, all_nodes)
    n_var = count(n -> n isa Variable, all_nodes)
    n_func = count(n -> n isa FunctionNode, all_nodes)
    
    unique_vars = length(collect_variables(node))
    unique_funcs = length(unique(collect_functions(node)))
    
    return (
        nodes = length(all_nodes),
        depth = tree_depth(node),
        terminals = n_const + n_var,
        functions = n_func,
        constants = n_const,
        variables = n_var,
        unique_vars = unique_vars,
        unique_funcs = unique_funcs,
    )
end
