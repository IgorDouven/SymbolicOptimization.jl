"""
    SymbolicOptimization

A Julia package for multi-objective symbolic optimization using genetic programming
and NSGA-II. Designed for discovering interpretable mathematical expressions that
optimize user-defined objectives.

# Features
- Flexible grammar specification (DSL, programmatic, or composable fragments)
- Support for typed and untyped expression trees
- Multi-objective optimization via NSGA-II
- Parallelization support (threads or distributed)
- Extensible genetic operators

# Quick Start (DSL)
```julia
using SymbolicOptimization

# Define and solve a symbolic regression problem
prob = symbolic_problem() do p
    variables!(p, :x)
    operators!(p, binary=[+, -, *, /], unary=[sin, cos, exp])
    constants!(p, range=-2.0:0.1:2.0)
    objectives!(p, :mse, :complexity)
    data!(p, X=X, y=y)
end

result = solve(prob)
println(best(result))
```

# Quick Start (JuMP-style API)
```julia
using SymbolicOptimization

m = SymbolicModel()
@variables(m, x, y)
@operators(m, binary=[+, -, *, /], unary=[sin, cos])
@objective(m, :mse)
@objective(m, :complexity)
@data(m, X=X, y=y)
optimize!(m)
println(best(m))
```

See the documentation for more details on grammar specification and advanced usage.
"""
module SymbolicOptimization

using Random
using Statistics
using Printf

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Core Types and Tree Utilities
# ═══════════════════════════════════════════════════════════════════════════════

include("types.jl")
include("tree_utils.jl")
include("printing.jl")

# Export core types
export AbstractNode, Constant, Variable, FunctionNode

# Export type utilities
export vartype, istyped, isconstant, isvariable, isfunction, isterminal
export children, arity

# Export tree utilities
export copy_tree, count_nodes, tree_depth
export flatten_tree, collect_variables, collect_constants, collect_functions
export replace_subtree, map_tree
export get_subtree_at_index, indexed_nodes
export random_subtree, random_subtree_index
export terminals, nonterminals, tree_size_stats

# Export printing
export node_to_string, node_to_latex
export print_tree, tree_to_string_block

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Grammar System
# ═══════════════════════════════════════════════════════════════════════════════

include("grammar/safe_ops.jl")
include("grammar/grammar.jl")
include("grammar/validation.jl")

# Export grammar types
export Grammar, OperatorSpec, VariableSpec, ConstantSpec
export ValidationResult

# Export grammar constructors and accessors
export is_typed, all_operators, all_variables, all_constants
export operators_by_name, operators_by_arity, operators_producing
export unary_operators, binary_operators, ternary_operators
export variables_of_type, constants_of_type
export has_type, has_operator, num_operators, num_variables
export sample_constant

# Export validation
export validate_grammar, validate_grammar!
export check_tree_validity, infer_type

# Export safe operations (useful for custom operators)
export safe_div, safe_pow, safe_log, safe_exp, safe_sqrt
export safe_mean, safe_sum, safe_std, safe_var
export sigmoid, relu, softplus, clamp01

# Export conditional operators (for piecewise functions like z measure)
export safe_step, safe_sign, safe_abs, safe_pos, safe_neg
export safe_max, safe_min, safe_ifelse, soft_ifelse

export SAFE_IMPLEMENTATIONS  # Dict for registering custom safe operator implementations

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Evaluation Engine
# ═══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra

include("evaluation/evaluate.jl")
include("evaluation/context.jl")

# Export evaluation types and functions
export EvalContext
export evaluate, evaluate_batch
export safe_evaluate, is_valid_on
export compile_tree
export has_operator, get_operator, with_operators, with_bindings

# Export context-aware evaluation system
export AbstractEvalContext, SimpleContext, VectorAggregatorContext
export resolve_variable, apply_operator, has_custom_operator, has_variable
export evaluate_with_context, safe_evaluate_with_context
export evaluate_sequential

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Genetic Operators
# ═══════════════════════════════════════════════════════════════════════════════

include("operators/generation.jl")
include("operators/mutation.jl")
include("operators/crossover.jl")
include("operators/simplification.jl")

# Export generation
export GenerationMethod, FullMethod, GrowMethod, HalfAndHalfMethod
export generate_tree, generate_population
export random_terminal, generate_random_subtree

# Export mutation
export mutate
export mutate_point, mutate_subtree, mutate_hoist
export mutate_constants, mutate_single_constant
export mutate_insert, mutate_delete

# Export crossover  
export crossover
export crossover_subtree, crossover_one_point
export crossover_uniform, crossover_size_fair

# Export simplification
export simplify, simplify_algebra, simplify_constants
export normalize_constants, clamp_constants

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: NSGA-II Optimization
# ═══════════════════════════════════════════════════════════════════════════════

include("optimization/types.jl")
include("optimization/nsga2_core.jl")
include("optimization/nsga2.jl")

# Export types
export Individual, ObjectiveFunction, NSGAIIConfig, NSGAIIResult

# Export objective constructors
export mse_objective, mae_objective, complexity_objective, depth_objective, custom_objective

# Export core algorithms
export dominates, nondominated_sort!, compute_crowding_distance!
export crowded_comparison, parsimony_comparison, tournament_select, environmental_select!

# Export main optimization functions
export optimize, curve_fitting, symbolic_regression
export get_best, get_pareto_front

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Pre-built Evaluators for Policy Problems
# ═══════════════════════════════════════════════════════════════════════════════

include("evaluators/evaluators.jl")
using .Evaluators

# Re-export evaluator functions
export discrimination_evaluator, sequential_evaluator, calibration_evaluator
export compute_auc, compute_brier, compute_log_score

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6b: Constraints for Theoretical Adequacy
# ═══════════════════════════════════════════════════════════════════════════════

include("constraints/constraints.jl")
using .Constraints

# Re-export constraint types and functions
export Constraint, ConstraintSet
export check_constraint, check_constraints, violation_rate
export directionality_constraint, logicality_constraint, symmetry_constraint
export monotonicity_constraint, boundary_constraint
export confirmation_measure_constraints

# ═══════════════════════════════════════════════════════════════════════════════
# Shared function interfaces (extended by both DSL and API modules)
# Defined here to avoid ambiguity when both submodules export the same name.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    add_objective!(target, args...; kwargs...)

Add an objective to a problem specification. Extended by DSL (for `SymbolicProblem`)
and API (for `SymbolicModel`).
"""
function add_objective! end

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6c: DSL (Domain-Specific Language) for Easy Use
# ═══════════════════════════════════════════════════════════════════════════════

include("dsl/dsl.jl")
using .DSL

# Re-export DSL types and functions
export SymbolicProblem, SymbolicResult, PolicyProblem
export variables!, operators!, constants!, add_function!, objectives!, data!, config!, mode!
export solve, best, pareto_front, evaluate_best
export symbolic_problem, policy_problem
export @symbolic_regression

# add_objective! is defined in this module and extended by both DSL and API
export add_objective!

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6d: JuMP-inspired API (@variables, @operators, @seed, optimize!, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

include("api/model.jl")
using .API

# Re-export the JuMP-like API
export SymbolicModel
export set_variables!, set_operators!, set_constants!, add_constraint!
export set_constraint_mode!, add_seed!, set_data!, set_config!
export optimize!
export @variables, @operators, @constants, @objective, @constraint, @seed, @data, @config, @grammar
export @problem
export expression_string, expression_latex, predict, history, raw_result
export objective_value, expr_to_tree
# Note: add_objective! already exported above (parent function extended by both DSL and API)
# Note: best() and pareto_front() already exported from DSL; API adds methods for SymbolicModel

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7: Pre-built Grammar Fragments (to be implemented)
# ═══════════════════════════════════════════════════════════════════════════════

# include("fragments/arithmetic.jl")
# include("fragments/transcendental.jl")
# include("fragments/vector_ops.jl")
# include("fragments/probability.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Symbolics.jl Integration (package extension)
# ═══════════════════════════════════════════════════════════════════════════════
# Functions defined here as stubs; implementations loaded when `using Symbolics`.

include("symbolics_interface.jl")

export to_symbolics, from_symbolics
export deep_simplify, simplified_string, simplified_latex
export simplify_piecewise, PiecewiseResult, complement_vars

end # module
