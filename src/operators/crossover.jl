# ═══════════════════════════════════════════════════════════════════════════════
# Crossover Operators
# ═══════════════════════════════════════════════════════════════════════════════
#
# Crossover strategies for combining expression trees.

using Random

# ───────────────────────────────────────────────────────────────────────────────
# Main Crossover Function
# ───────────────────────────────────────────────────────────────────────────────

"""
    crossover(parent1::AbstractNode, parent2::AbstractNode; kwargs...) -> Tuple{AbstractNode, AbstractNode}

Perform crossover between two parent trees, producing two offspring.

# Keyword Arguments
- `rng::AbstractRNG = Random.GLOBAL_RNG`: Random number generator
- `max_depth::Int = 10`: Maximum depth of resulting trees
- `internal_prob::Float64 = 0.9`: Probability of selecting internal (non-terminal) nodes

Returns a tuple of two new trees (originals are not modified).
"""
function crossover(parent1::AbstractNode, parent2::AbstractNode;
                   rng::AbstractRNG = Random.GLOBAL_RNG,
                   max_depth::Int = 10,
                   internal_prob::Float64 = 0.9)
    
    return crossover_subtree(parent1, parent2; rng=rng, max_depth=max_depth, internal_prob=internal_prob)
end

# ───────────────────────────────────────────────────────────────────────────────
# Subtree Crossover
# ───────────────────────────────────────────────────────────────────────────────

"""
    crossover_subtree(parent1::AbstractNode, parent2::AbstractNode; kwargs...) -> Tuple{AbstractNode, AbstractNode}

Standard subtree crossover (Koza-style).

Selects a random subtree from each parent and swaps them.
"""
function crossover_subtree(parent1::AbstractNode, parent2::AbstractNode;
                           rng::AbstractRNG = Random.GLOBAL_RNG,
                           max_depth::Int = 10,
                           internal_prob::Float64 = 0.9)
    
    # Select crossover points
    point1, depth1 = _select_crossover_point(parent1, internal_prob, rng)
    point2, depth2 = _select_crossover_point(parent2, internal_prob, rng)
    
    # Get the subtrees
    subtree1 = copy_tree(point1)
    subtree2 = copy_tree(point2)
    
    # Create offspring by swapping subtrees
    child1 = _replace_node(copy_tree(parent1), point1, subtree2)
    child2 = _replace_node(copy_tree(parent2), point2, subtree1)
    
    # Enforce depth limit by replacing with terminal if too deep
    if tree_depth(child1) > max_depth
        child1 = _truncate_tree(child1, max_depth)
    end
    if tree_depth(child2) > max_depth
        child2 = _truncate_tree(child2, max_depth)
    end
    
    return (child1, child2)
end

"""
    crossover_one_point(parent1::AbstractNode, parent2::AbstractNode; kwargs...) -> Tuple{AbstractNode, AbstractNode}

One-point crossover: only the first child is produced by swapping subtrees.
The second child is the other swap direction.
"""
crossover_one_point = crossover_subtree  # Alias, same implementation

# ───────────────────────────────────────────────────────────────────────────────
# Uniform Crossover
# ───────────────────────────────────────────────────────────────────────────────

"""
    crossover_uniform(parent1::AbstractNode, parent2::AbstractNode; kwargs...) -> Tuple{AbstractNode, AbstractNode}

Uniform crossover: at each node position, randomly choose from either parent.

Note: This only works well when parents have similar structure.
Falls back to subtree crossover if structures are too different.
"""
function crossover_uniform(parent1::AbstractNode, parent2::AbstractNode;
                           rng::AbstractRNG = Random.GLOBAL_RNG,
                           max_depth::Int = 10,
                           p_swap::Float64 = 0.5)
    
    child1 = _uniform_crossover_node(parent1, parent2, p_swap, rng)
    child2 = _uniform_crossover_node(parent2, parent1, p_swap, rng)
    
    # Enforce depth limits
    if tree_depth(child1) > max_depth
        child1 = _truncate_tree(child1, max_depth)
    end
    if tree_depth(child2) > max_depth
        child2 = _truncate_tree(child2, max_depth)
    end
    
    return (child1, child2)
end

function _uniform_crossover_node(node1::AbstractNode, node2::AbstractNode, 
                                  p_swap::Float64, rng::AbstractRNG)
    # With probability p_swap, take from node2 instead
    if rand(rng) < p_swap
        return copy_tree(node2)
    end
    
    # Otherwise, take from node1 but recurse on children
    if node1 isa FunctionNode && node2 isa FunctionNode && 
       node1.func == node2.func && length(node1.children) == length(node2.children)
        # Same structure - can do uniform crossover on children
        new_children = AbstractNode[]
        for (c1, c2) in zip(node1.children, node2.children)
            push!(new_children, _uniform_crossover_node(c1, c2, p_swap, rng))
        end
        return FunctionNode(node1.func, new_children)
    else
        # Different structure - just take one or the other
        return copy_tree(node1)
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Size-Fair Crossover
# ───────────────────────────────────────────────────────────────────────────────

"""
    crossover_size_fair(parent1::AbstractNode, parent2::AbstractNode; kwargs...) -> Tuple{AbstractNode, AbstractNode}

Size-fair crossover: tries to select subtrees of similar size to avoid bloat.
"""
function crossover_size_fair(parent1::AbstractNode, parent2::AbstractNode;
                             rng::AbstractRNG = Random.GLOBAL_RNG,
                             max_depth::Int = 10,
                             internal_prob::Float64 = 0.9,
                             size_tolerance::Int = 5)
    
    # Get all nodes with sizes
    nodes1 = [(n, count_nodes(n)) for n in flatten_tree(parent1)]
    nodes2 = [(n, count_nodes(n)) for n in flatten_tree(parent2)]
    
    # Separate internal and terminal
    internal1 = filter(x -> !isterminal(x[1]), nodes1)
    internal2 = filter(x -> !isterminal(x[1]), nodes2)
    terminal1 = filter(x -> isterminal(x[1]), nodes1)
    terminal2 = filter(x -> isterminal(x[1]), nodes2)
    
    # Select from internal or terminal based on probability
    if !isempty(internal1) && rand(rng) < internal_prob
        candidates1 = internal1
    else
        candidates1 = isempty(terminal1) ? nodes1 : terminal1
    end
    
    # Select first crossover point
    point1, size1 = rand(rng, candidates1)
    
    # Find matching-size node in parent2
    if !isempty(internal2) && rand(rng) < internal_prob
        candidates2 = internal2
    else
        candidates2 = isempty(terminal2) ? nodes2 : terminal2
    end
    
    # Filter by size similarity
    size_matched = filter(x -> abs(x[2] - size1) <= size_tolerance, candidates2)
    
    if isempty(size_matched)
        size_matched = candidates2
    end
    
    point2, _ = rand(rng, size_matched)
    
    # Perform the swap
    subtree1 = copy_tree(point1)
    subtree2 = copy_tree(point2)
    
    child1 = _replace_node(copy_tree(parent1), point1, subtree2)
    child2 = _replace_node(copy_tree(parent2), point2, subtree1)
    
    # Enforce depth limit
    if tree_depth(child1) > max_depth
        child1 = _truncate_tree(child1, max_depth)
    end
    if tree_depth(child2) > max_depth
        child2 = _truncate_tree(child2, max_depth)
    end
    
    return (child1, child2)
end

# ───────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────────────────────

"""
Select a crossover point in a tree.
"""
function _select_crossover_point(tree::AbstractNode, internal_prob::Float64, rng::AbstractRNG)
    nodes = flatten_tree(tree)
    
    # Separate internal and terminal nodes
    internal = filter(n -> !isterminal(n), nodes)
    terminal = filter(n -> isterminal(n), nodes)
    
    # Choose internal or terminal based on probability
    if !isempty(internal) && rand(rng) < internal_prob
        point = rand(rng, internal)
    elseif !isempty(terminal)
        point = rand(rng, terminal)
    else
        point = rand(rng, nodes)
    end
    
    # Calculate depth of this point in the tree
    depth = _node_depth(tree, point)
    
    return (point, depth)
end

"""
Find the depth of a specific node in a tree.
"""
function _node_depth(tree::AbstractNode, target::AbstractNode, current_depth::Int = 1)
    if tree === target
        return current_depth
    end
    
    if tree isa FunctionNode
        for child in tree.children
            d = _node_depth(child, target, current_depth + 1)
            if d > 0
                return d
            end
        end
    end
    
    return 0
end

# Note: _replace_node is defined in mutation.jl and reused here

"""
Truncate a tree to the maximum depth by replacing deep subtrees with terminals.
"""
function _truncate_tree(tree::AbstractNode, max_depth::Int, current_depth::Int = 1)
    if current_depth >= max_depth
        # At max depth - convert to terminal
        if tree isa FunctionNode
            # Return a child (simplest truncation)
            return _truncate_tree(tree.children[1], max_depth, current_depth)
        else
            return copy_tree(tree)
        end
    end
    
    if tree isa FunctionNode
        new_children = AbstractNode[]
        for child in tree.children
            push!(new_children, _truncate_tree(child, max_depth, current_depth + 1))
        end
        return FunctionNode(tree.func, new_children)
    end
    
    return copy_tree(tree)
end
