#=
Basic Example: Working with Expression Trees
=============================================

This example demonstrates the core functionality of SymbolicOptimization.jl
Phase 1: creating, manipulating, and printing expression trees.

Run this from the package directory:
    julia --project=. examples/basic_trees.jl
=#

using SymbolicOptimization

println("="^60)
println("SymbolicOptimization.jl - Basic Tree Operations")
println("="^60)

# ─────────────────────────────────────────────────────────────
# Creating Nodes
# ─────────────────────────────────────────────────────────────

println("\n1. Creating Basic Nodes")
println("-"^40)

# Constants
c1 = Constant(3.14)
c2 = Constant(2)
println("Constants: ", c1.value, ", ", c2.value)

# Variables (untyped)
x = Variable(:x)
y = Variable(:y)
println("Untyped variables: ", x.name, ", ", y.name)

# Variables (typed - for grammars with type systems)
ps = Variable(:ps, :Vector)
n = Variable(:n, :Scalar)
println("Typed variables: ", ps.name, " :: ", vartype(ps), ", ", n.name, " :: ", vartype(n))

# ─────────────────────────────────────────────────────────────
# Building Expression Trees
# ─────────────────────────────────────────────────────────────

println("\n2. Building Expression Trees")
println("-"^40)

# Simple: x + 1
simple = FunctionNode(:+, x, Constant(1.0))
println("Simple tree: ", node_to_string(simple))

# Medium: (x + 1) * y  
medium = FunctionNode(:*, simple, y)
println("Medium tree: ", node_to_string(medium))

# Complex: sin((x + 1) * y) + 2
complex = FunctionNode(:+, 
    FunctionNode(:sin, medium),
    Constant(2.0)
)
println("Complex tree: ", node_to_string(complex))

# ─────────────────────────────────────────────────────────────
# Tree Statistics
# ─────────────────────────────────────────────────────────────

println("\n3. Tree Statistics")
println("-"^40)

println("Simple tree:")
println("  Nodes: ", count_nodes(simple))
println("  Depth: ", tree_depth(simple))
println("  Variables: ", collect_variables(simple))

println("\nComplex tree:")
stats = tree_size_stats(complex)
println("  Nodes: ", stats.nodes)
println("  Depth: ", stats.depth)
println("  Variables: ", stats.unique_vars, " unique")
println("  Functions: ", stats.unique_funcs, " unique")

# ─────────────────────────────────────────────────────────────
# Tree Traversal
# ─────────────────────────────────────────────────────────────

println("\n4. Tree Traversal")
println("-"^40)

println("All nodes (pre-order):")
for (i, node) in indexed_nodes(complex)
    local type_str = if isconstant(node)
        "Constant($(node.value))"
    elseif isvariable(node)
        "Variable(:$(node.name))"
    else
        "Function(:$(node.func))"
    end
    println("  [$i] $type_str")
end

# ─────────────────────────────────────────────────────────────
# Tree Manipulation
# ─────────────────────────────────────────────────────────────

println("\n5. Tree Manipulation")
println("-"^40)

# Copy
copied = copy_tree(complex)
println("Copied tree: ", node_to_string(copied))
println("Same structure: ", copied == complex)
println("Same object: ", copied === complex)

# Replace subtree
modified = replace_subtree(simple, x, Constant(5.0))
println("\nOriginal: ", node_to_string(simple))
println("After replacing x with 5: ", node_to_string(modified))

# Map tree (double all constants)
doubled = map_tree(simple) do node
    if isconstant(node)
        Constant(node.value * 2)
    else
        node
    end
end
println("\nOriginal: ", node_to_string(simple))
println("Constants doubled: ", node_to_string(doubled))

# ─────────────────────────────────────────────────────────────
# Pretty Printing
# ─────────────────────────────────────────────────────────────

println("\n6. Pretty Printing")
println("-"^40)

# Standard string
println("String representation:")
println("  ", node_to_string(complex))

# LaTeX
println("\nLaTeX representation:")
println("  ", node_to_latex(complex))

# Hierarchical tree view
println("\nTree structure:")
print_tree(complex)

# ─────────────────────────────────────────────────────────────
# Advanced: Building an Aggregator Expression
# ─────────────────────────────────────────────────────────────

println("\n7. Example: Probability Aggregator")
println("-"^40)

# Build: clamp01(mean(ps^var(log(ps))))
# This is the "adaptive power mean" from crowd wisdom research

ps = Variable(:ps, :Vector)

adaptive_power_mean = FunctionNode(:clamp01,
    FunctionNode(:mean,
        FunctionNode(:.^, 
            ps,
            FunctionNode(:var,
                FunctionNode(:log, ps)
            )
        )
    )
)

println("Adaptive power mean aggregator:")
println("  String: ", node_to_string(adaptive_power_mean))
println("  LaTeX:  ", node_to_latex(adaptive_power_mean))
println("  Nodes:  ", count_nodes(adaptive_power_mean))
println("\nTree structure:")
print_tree(adaptive_power_mean)

# ─────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("Phase 1 complete! Next: Grammar specification (Phase 2)")
println("="^60)
