#=
Grammar System Example
======================

This example demonstrates how to create and use grammars in SymbolicOptimization.jl.

Run this from the package directory:
    julia --project=. examples/grammar_example.jl
=#

using SymbolicOptimization
using Statistics: mean, sum, std, var

println("="^60)
println("SymbolicOptimization.jl - Grammar System")
println("="^60)

# ─────────────────────────────────────────────────────────────
# 1. Simple (Untyped) Grammar
# ─────────────────────────────────────────────────────────────

println("\n1. Simple Grammar (like SymbolicRegression.jl)")
println("-"^50)

# This style is familiar to SymbolicRegression.jl users
simple_grammar = Grammar(
    binary_operators = [+, -, *, /],
    unary_operators = [sin, cos, exp, log],
    variables = [:x, :y],
    constant_range = (-2.0, 2.0),
)

println(simple_grammar)
println()

# Access operators
println("Unary operators: ", [op.name for op in unary_operators(simple_grammar)])
println("Binary operators: ", [op.name for op in binary_operators(simple_grammar)])
println()

# Sample some constants
println("Sample constants: ", [round(sample_constant(simple_grammar), digits=2) for _ in 1:5])

# ─────────────────────────────────────────────────────────────
# 2. Typed Grammar (for Vector/Scalar Operations)
# ─────────────────────────────────────────────────────────────

println("\n2. Typed Grammar (Vector ↔ Scalar)")
println("-"^50)

# This is useful for aggregation/pooling problems
typed_grammar = Grammar(
    types = [:Scalar, :Vector],
    
    variables = [
        :ps => :Vector,    # Vector of probabilities
        :n => :Scalar,     # Scalar (e.g., count)
    ],
    
    operators = [
        # Reductions: Vector → Scalar
        (mean, [:Vector] => :Scalar),
        (sum, [:Vector] => :Scalar),
        (:std, [:Vector] => :Scalar),
        (:var, [:Vector] => :Scalar),
        (:maximum, [:Vector] => :Scalar),
        (:minimum, [:Vector] => :Scalar),
        
        # Scalar arithmetic
        (+, [:Scalar, :Scalar] => :Scalar),
        (-, [:Scalar, :Scalar] => :Scalar),
        (*, [:Scalar, :Scalar] => :Scalar),
        (/, [:Scalar, :Scalar] => :Scalar),
        
        # Scalar functions
        (exp, [:Scalar] => :Scalar),
        (log, [:Scalar] => :Scalar),
        (:sigmoid, [:Scalar] => :Scalar),
        (:clamp01, [:Scalar] => :Scalar),
        
        # Element-wise: (Vector, Scalar) → Vector
        (Symbol(".^"), [:Vector, :Scalar] => :Vector),
        (Symbol(".*"), [:Vector, :Scalar] => :Vector),
        
        # Element-wise: Vector → Vector  
        (:ew_log, [:Vector] => :Vector),
        (:ew_exp, [:Vector] => :Vector),
    ],
    
    constant_types = [:Scalar],
    constant_range = (-2.0, 2.0),
    output_type = :Scalar,
)

# Show detailed grammar
println(typed_grammar)

# ─────────────────────────────────────────────────────────────
# 3. Grammar Validation
# ─────────────────────────────────────────────────────────────

println("\n3. Grammar Validation")
println("-"^50)

result = validate_grammar(typed_grammar)
println("Valid: ", result.valid)
println("Errors: ", result.errors)
println("Warnings: ", result.warnings)

# ─────────────────────────────────────────────────────────────
# 4. Building Trees with Grammar
# ─────────────────────────────────────────────────────────────

println("\n4. Building and Validating Trees")
println("-"^50)

# Build a valid tree for simple grammar: sin(x + 1.5)
tree1 = FunctionNode(:sin, 
    FunctionNode(:+, Variable(:x), Constant(1.5))
)
println("Tree 1: ", node_to_string(tree1))

valid, msg = check_tree_validity(tree1, simple_grammar)
println("Valid in simple grammar: $valid")

# Build a valid tree for typed grammar: clamp01(mean(ps .^ 2))
tree2 = FunctionNode(:clamp01,
    FunctionNode(:mean,
        FunctionNode(Symbol(".^"), Variable(:ps), Constant(2.0))
    )
)
println("\nTree 2: ", node_to_string(tree2))

valid2, msg2 = check_tree_validity(tree2, typed_grammar)
println("Valid in typed grammar: $valid2")

# Try an invalid tree (unknown operator)
tree3 = FunctionNode(:unknown_op, Variable(:x))
valid3, msg3 = check_tree_validity(tree3, simple_grammar)
println("\nTree 3 (invalid): ", node_to_string(tree3))
println("Valid: $valid3, Message: $msg3")

# ─────────────────────────────────────────────────────────────
# 5. Type Inference
# ─────────────────────────────────────────────────────────────

println("\n5. Type Inference (Typed Grammar)")
println("-"^50)

# Infer types of subexpressions
subexpr1 = Variable(:ps)
type1 = infer_type(subexpr1, typed_grammar)
println("Type of 'ps': $type1")

subexpr2 = FunctionNode(Symbol(".^"), Variable(:ps), Constant(2.0))
type2 = infer_type(subexpr2, typed_grammar)
println("Type of 'ps .^ 2': $type2")

subexpr3 = FunctionNode(:mean, subexpr2)
type3 = infer_type(subexpr3, typed_grammar)
println("Type of 'mean(ps .^ 2)': $type3")

# ─────────────────────────────────────────────────────────────
# 6. Querying Grammar
# ─────────────────────────────────────────────────────────────

println("\n6. Querying Grammar")
println("-"^50)

# Operators that produce Scalar
scalar_ops = operators_producing(typed_grammar, :Scalar)
println("Operators producing Scalar: ", [op.name for op in scalar_ops])

# Operators that produce Vector
vector_ops = operators_producing(typed_grammar, :Vector)
println("Operators producing Vector: ", [op.name for op in vector_ops])

# Variables of each type
println("Scalar variables: ", [v.name for v in variables_of_type(typed_grammar, :Scalar)])
println("Vector variables: ", [v.name for v in variables_of_type(typed_grammar, :Vector)])

# ─────────────────────────────────────────────────────────────
# 7. Custom Complexity
# ─────────────────────────────────────────────────────────────

println("\n7. Custom Complexity Weights")
println("-"^50)

complex_grammar = Grammar(
    binary_operators = [+, -, *, /, (^)],
    unary_operators = [sin, cos, exp, log],
    variables = [:x],
    
    # Make some operators more "expensive"
    complexity_of_operators = Dict(
        (^) => 3.0,    # Power is complex
        (/) => 2.0,    # Division is moderately complex
        exp => 2.0,
        log => 2.0,
    ),
    complexity_of_variables = 0.5,
    complexity_of_constants = 1.0,
)

println("Operator complexities:")
for op in all_operators(complex_grammar)
    println("  $(op.name): $(op.complexity)")
end

# ─────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("Phase 2 complete! Next: Tree generation and evaluation")
println("="^60)
