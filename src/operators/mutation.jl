# ═══════════════════════════════════════════════════════════════════════════════
# Mutation Operators
# ═══════════════════════════════════════════════════════════════════════════════
#
# Various mutation strategies for evolving expression trees.

using Random

# ───────────────────────────────────────────────────────────────────────────────
# Main Mutation Function
# ───────────────────────────────────────────────────────────────────────────────

"""
    mutate(tree::AbstractNode, grammar::Grammar; kwargs...) -> AbstractNode

Apply mutation to an expression tree.

# Keyword Arguments
- `rng::AbstractRNG = Random.GLOBAL_RNG`: Random number generator
- `p_point::Float64 = 0.3`: Probability of point mutation
- `p_subtree::Float64 = 0.3`: Probability of subtree mutation
- `p_hoist::Float64 = 0.1`: Probability of hoist mutation
- `p_constant::Float64 = 0.2`: Probability of constant perturbation
- `p_insert::Float64 = 0.05`: Probability of insert mutation
- `p_delete::Float64 = 0.05`: Probability of delete mutation
- `max_depth::Int = 4`: Maximum depth for new subtrees
- `constant_std::Float64 = 0.5`: Standard deviation for constant perturbation

Returns a new mutated tree (original is not modified).
"""
function mutate(tree::AbstractNode, grammar::Grammar;
                rng::AbstractRNG = Random.GLOBAL_RNG,
                p_point::Float64 = 0.3,
                p_subtree::Float64 = 0.3,
                p_hoist::Float64 = 0.1,
                p_constant::Float64 = 0.2,
                p_insert::Float64 = 0.05,
                p_delete::Float64 = 0.05,
                max_depth::Int = 4,
                constant_std::Float64 = 0.5)
    
    # Normalize probabilities
    total = p_point + p_subtree + p_hoist + p_constant + p_insert + p_delete
    r = rand(rng) * total
    
    if r < p_point
        return mutate_point(tree, grammar; rng=rng)
    elseif r < p_point + p_subtree
        return mutate_subtree(tree, grammar; rng=rng, max_depth=max_depth)
    elseif r < p_point + p_subtree + p_hoist
        return mutate_hoist(tree; rng=rng)
    elseif r < p_point + p_subtree + p_hoist + p_constant
        return mutate_constants(tree; rng=rng, std=constant_std)
    elseif r < p_point + p_subtree + p_hoist + p_constant + p_insert
        return mutate_insert(tree, grammar; rng=rng)
    else
        return mutate_delete(tree; rng=rng)
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Point Mutation
# ───────────────────────────────────────────────────────────────────────────────

"""
    mutate_point(tree::AbstractNode, grammar::Grammar; rng=Random.GLOBAL_RNG) -> AbstractNode

Replace a random node with another node of the same arity/type.

- Function nodes: Replace with another function of same arity
- Variables: Replace with another variable
- Constants: Replace with a new random constant
"""
function mutate_point(tree::AbstractNode, grammar::Grammar;
                      rng::AbstractRNG = Random.GLOBAL_RNG)
    tree = copy_tree(tree)
    nodes = flatten_tree(tree)
    
    if isempty(nodes)
        return tree
    end
    
    # Select a random node to mutate
    idx = rand(rng, 1:length(nodes))
    target = nodes[idx]
    
    # Create replacement
    replacement = _point_mutate_node(target, grammar, rng)
    
    if replacement === target
        return tree
    end
    
    # Find and replace in tree
    return _replace_node(tree, target, replacement)
end

function _point_mutate_node(node::Constant, grammar::Grammar, rng::AbstractRNG)
    # Replace with new constant
    Constant(sample_constant(grammar))
end

function _point_mutate_node(node::Variable, grammar::Grammar, rng::AbstractRNG)
    # Replace with another variable (possibly of same type)
    vars = if grammar.is_typed && istyped(node)
        vs = variables_of_type(grammar, vartype(node))
        isempty(vs) ? grammar.variables : vs
    else
        grammar.variables
    end
    
    if length(vars) <= 1
        return node  # No other options
    end
    
    # Pick a different variable
    new_var = rand(rng, vars)
    attempts = 0
    while new_var.name == node.name && attempts < 10
        new_var = rand(rng, vars)
        attempts += 1
    end
    
    if grammar.is_typed && new_var.type != :Any
        return Variable(new_var.name, new_var.type)
    else
        return Variable(new_var.name)
    end
end

function _point_mutate_node(node::FunctionNode, grammar::Grammar, rng::AbstractRNG)
    # Replace with another function of same arity
    same_arity_ops = operators_by_arity(grammar, length(node.children))
    
    if length(same_arity_ops) <= 1
        return node  # No other options
    end
    
    # Pick a different operator
    new_op = rand(rng, same_arity_ops)
    attempts = 0
    while new_op.name == node.func && attempts < 10
        new_op = rand(rng, same_arity_ops)
        attempts += 1
    end
    
    return FunctionNode(new_op.name, copy_tree.(node.children))
end

# ───────────────────────────────────────────────────────────────────────────────
# Subtree Mutation
# ───────────────────────────────────────────────────────────────────────────────

"""
    mutate_subtree(tree::AbstractNode, grammar::Grammar; rng=Random.GLOBAL_RNG, max_depth=4) -> AbstractNode

Replace a random subtree with a newly generated subtree.
"""
function mutate_subtree(tree::AbstractNode, grammar::Grammar;
                        rng::AbstractRNG = Random.GLOBAL_RNG,
                        max_depth::Int = 4)
    tree = copy_tree(tree)
    nodes = flatten_tree(tree)
    
    if isempty(nodes)
        return tree
    end
    
    # Select a random subtree
    idx = rand(rng, 1:length(nodes))
    target = nodes[idx]
    
    # Determine target type for typed grammars
    target_type = if grammar.is_typed && target isa Variable && istyped(target)
        vartype(target)
    else
        :Any
    end
    
    # Generate replacement subtree
    new_subtree = generate_tree(grammar;
                                method = GrowMethod(),
                                min_depth = 1,
                                max_depth = max_depth,
                                target_type = target_type,
                                rng = rng)
    
    return _replace_node(tree, target, new_subtree)
end

# ───────────────────────────────────────────────────────────────────────────────
# Hoist Mutation
# ───────────────────────────────────────────────────────────────────────────────

"""
    mutate_hoist(tree::AbstractNode; rng=Random.GLOBAL_RNG) -> AbstractNode

Replace the tree with one of its subtrees, effectively "hoisting" it up.
Tends to simplify/shrink trees.
"""
function mutate_hoist(tree::AbstractNode; rng::AbstractRNG = Random.GLOBAL_RNG)
    if isterminal(tree)
        return copy_tree(tree)
    end
    
    # Get all non-root subtrees
    nodes = flatten_tree(tree)
    subtrees = filter(n -> n !== tree, nodes)
    
    if isempty(subtrees)
        return copy_tree(tree)
    end
    
    # Select and return a random subtree
    return copy_tree(rand(rng, subtrees))
end

# ───────────────────────────────────────────────────────────────────────────────
# Constant Perturbation
# ───────────────────────────────────────────────────────────────────────────────

"""
    mutate_constants(tree::AbstractNode; rng=Random.GLOBAL_RNG, std=0.5) -> AbstractNode

Perturb all constants in the tree by adding Gaussian noise.
"""
function mutate_constants(tree::AbstractNode;
                          rng::AbstractRNG = Random.GLOBAL_RNG,
                          std::Float64 = 0.5)
    map_tree(tree) do node
        if node isa Constant
            Constant(node.value + randn(rng) * std)
        else
            node
        end
    end
end

"""
    mutate_single_constant(tree::AbstractNode; rng=Random.GLOBAL_RNG, std=0.5) -> AbstractNode

Perturb a single randomly selected constant.
"""
function mutate_single_constant(tree::AbstractNode;
                                rng::AbstractRNG = Random.GLOBAL_RNG,
                                std::Float64 = 0.5)
    tree = copy_tree(tree)
    constants = collect_constants(tree)
    
    if isempty(constants)
        return tree
    end
    
    # Find constant nodes
    const_nodes = filter(n -> n isa Constant, flatten_tree(tree))
    
    if isempty(const_nodes)
        return tree
    end
    
    # Select one to mutate
    target = rand(rng, const_nodes)
    replacement = Constant(target.value + randn(rng) * std)
    
    return _replace_node(tree, target, replacement)
end

# ───────────────────────────────────────────────────────────────────────────────
# Insert Mutation
# ───────────────────────────────────────────────────────────────────────────────

"""
    mutate_insert(tree::AbstractNode, grammar::Grammar; rng=Random.GLOBAL_RNG) -> AbstractNode

Insert a new operator above a random node, making the original node a child.
Tends to grow trees.
"""
function mutate_insert(tree::AbstractNode, grammar::Grammar;
                       rng::AbstractRNG = Random.GLOBAL_RNG)
    tree = copy_tree(tree)
    nodes = flatten_tree(tree)
    
    if isempty(nodes)
        return tree
    end
    
    # Select insertion point
    idx = rand(rng, 1:length(nodes))
    target = nodes[idx]
    
    # Get operators with arity >= 1
    valid_ops = filter(op -> op.arity >= 1, grammar.operators)
    
    if isempty(valid_ops)
        return tree
    end
    
    # Select an operator
    op = rand(rng, valid_ops)
    
    # Create new node with target as first child
    children = AbstractNode[copy_tree(target)]
    
    # Generate remaining children
    for i in 2:op.arity
        input_type = if !isempty(op.input_types) && i <= length(op.input_types)
            op.input_types[i]
        else
            :Any
        end
        child = random_terminal(grammar; target_type=input_type, rng=rng)
        push!(children, child)
    end
    
    new_node = FunctionNode(op.name, children)
    
    return _replace_node(tree, target, new_node)
end

# ───────────────────────────────────────────────────────────────────────────────
# Delete Mutation
# ───────────────────────────────────────────────────────────────────────────────

"""
    mutate_delete(tree::AbstractNode; rng=Random.GLOBAL_RNG) -> AbstractNode

Delete a function node by replacing it with one of its children.
Tends to shrink trees.
"""
function mutate_delete(tree::AbstractNode; rng::AbstractRNG = Random.GLOBAL_RNG)
    if isterminal(tree)
        return copy_tree(tree)
    end
    
    tree = copy_tree(tree)
    
    # Find function nodes
    func_nodes = filter(n -> n isa FunctionNode, flatten_tree(tree))
    
    if isempty(func_nodes)
        return tree
    end
    
    # Select a function node to delete
    target = rand(rng, func_nodes)
    
    # Replace with one of its children
    replacement = copy_tree(rand(rng, target.children))
    
    if target === tree
        return replacement
    end
    
    return _replace_node(tree, target, replacement)
end

# ───────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────────────────────

"""
Replace a specific node in a tree with a new node.
"""
function _replace_node(tree::AbstractNode, target::AbstractNode, replacement::AbstractNode)
    if tree === target
        return replacement
    end
    
    if tree isa FunctionNode
        new_children = AbstractNode[]
        for child in tree.children
            if child === target
                push!(new_children, replacement)
            else
                push!(new_children, _replace_node(child, target, replacement))
            end
        end
        return FunctionNode(tree.func, new_children)
    end
    
    return tree
end
