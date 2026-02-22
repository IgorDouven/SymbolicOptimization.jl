#=
DSL (Domain-Specific Language) for Symbolic Optimization
=========================================================

Provides a user-friendly macro-based interface for defining and running
symbolic optimization problems without dealing with low-level details.

Example usage:

```julia
using SymbolicOptimization

# Simple curve fitting
result = @optimize begin
    @variables x
    @operators +, -, *, /
    @objective mse(pred, y) = mean((pred .- y).^2)
    @data X = my_x_data, y = my_y_data
end

# Probability aggregation with custom functions  
result = @optimize begin
    @variables p1, p2, p3, p4, p5
    @operators +, -, *, /
    @constants 0.0:2.0
    
    @function geometric_mean(ps) = exp(mean(log.(clamp.(ps, 1e-10, 1))))
    
    @objective brier(pred, truth) = mean((pred .- truth).^2)
    @objective complexity(expr) = count_nodes(expr)
    
    @data probs = probability_matrix, truth = ground_truth
    
    @config population=200, generations=100
end
```
=#

module DSL

using Statistics: mean, std
using Random: MersenneTwister

# Import from parent module (will be set up in SymbolicOptimization.jl)
import ..Grammar, ..NSGAIIConfig, ..optimize, ..mse_objective, ..complexity_objective, ..custom_objective
import ..get_best, ..get_pareto_front, ..node_to_string, ..evaluate, ..Individual, ..ObjectiveFunction
import ..count_nodes
import ..Variable, ..Constant, ..FunctionNode
import ..check_constraints
import ..add_objective!  # Defined in parent; DSL extends it for SymbolicProblem

export @symbolic_regression, @optimize_expression, SymbolicProblem, solve

# ─────────────────────────────────────────────────────────────────────────────
# Tree-building operators for seed_formulas
# ─────────────────────────────────────────────────────────────────────────────
# These allow users to write `v.pH_E - v.pE_notH` and get a tree structure

# Helper to convert function to symbol
func_to_symbol(f::Function) = Symbol(f)

# Define arithmetic on tree nodes
Base.:+(a::Union{Variable, Constant, FunctionNode}, b::Union{Variable, Constant, FunctionNode}) = FunctionNode(:+, [a, b])
Base.:-(a::Union{Variable, Constant, FunctionNode}, b::Union{Variable, Constant, FunctionNode}) = FunctionNode(:-, [a, b])
Base.:*(a::Union{Variable, Constant, FunctionNode}, b::Union{Variable, Constant, FunctionNode}) = FunctionNode(:*, [a, b])
Base.:/(a::Union{Variable, Constant, FunctionNode}, b::Union{Variable, Constant, FunctionNode}) = FunctionNode(:/, [a, b])

# Allow mixing with numbers (constants)
Base.:+(a::Union{Variable, Constant, FunctionNode}, b::Real) = FunctionNode(:+, [a, Constant(Float64(b))])
Base.:+(a::Real, b::Union{Variable, Constant, FunctionNode}) = FunctionNode(:+, [Constant(Float64(a)), b])
Base.:-(a::Union{Variable, Constant, FunctionNode}, b::Real) = FunctionNode(:-, [a, Constant(Float64(b))])
Base.:-(a::Real, b::Union{Variable, Constant, FunctionNode}) = FunctionNode(:-, [Constant(Float64(a)), b])
Base.:*(a::Union{Variable, Constant, FunctionNode}, b::Real) = FunctionNode(:*, [a, Constant(Float64(b))])
Base.:*(a::Real, b::Union{Variable, Constant, FunctionNode}) = FunctionNode(:*, [Constant(Float64(a)), b])
Base.:/(a::Union{Variable, Constant, FunctionNode}, b::Real) = FunctionNode(:/, [a, Constant(Float64(b))])
Base.:/(a::Real, b::Union{Variable, Constant, FunctionNode}) = FunctionNode(:/, [Constant(Float64(a)), b])

# Unary minus
Base.:-(a::Union{Variable, Constant, FunctionNode}) = FunctionNode(:*, [Constant(-1.0), a])

"""
    build_seed_tree(formula_fn, variables) -> tree

Build an expression tree from a seed formula function.
The function receives a NamedTuple of variable nodes.
"""
function build_seed_tree(formula_fn::Function, variables::Vector{Symbol})
    # Create variable nodes
    var_nodes = Dict(v => Variable(v) for v in variables)
    
    # Create a NamedTuple so users can write v.pH_E syntax
    var_tuple = NamedTuple{Tuple(variables)}(Tuple(var_nodes[v] for v in variables))
    
    # Call the formula function to build the tree
    return formula_fn(var_tuple)
end

# ─────────────────────────────────────────────────────────────────────────────
# Problem Definition Struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    SymbolicProblem

A high-level specification of a symbolic optimization problem.
Created by helper functions, then solved with `solve()`.
"""
mutable struct SymbolicProblem
    # Grammar specification
    variables::Vector{Symbol}
    binary_operators::Vector{Function}
    unary_operators::Vector{Function}
    custom_functions::Dict{Symbol, Function}
    constant_range::Tuple{Float64, Float64}
    constant_prob::Float64
    
    # Objectives
    objectives::Vector{Symbol}  # :mse, :complexity, :brier, or custom
    custom_objectives::Dict{Symbol, Function}
    
    # Data
    X::Union{Nothing, AbstractMatrix}
    y::Union{Nothing, AbstractVector}
    extra_data::Dict{Symbol, Any}
    
    # Evaluation mode
    # :regression - standard row-by-row (X[i,:] -> prediction, compare to y[i])
    # :aggregation - variables are columns, output is aggregated prediction per row
    eval_mode::Symbol
    
    # Configuration
    population_size::Int
    max_generations::Int
    max_depth::Int
    max_nodes::Int
    validation_fraction::Float64
    seed::Union{Nothing, Int}
    verbose::Bool
end

# Default constructor
function SymbolicProblem()
    SymbolicProblem(
        Symbol[],                          # variables
        Function[+, -, *, /],              # binary_operators
        Function[],                        # unary_operators
        Dict{Symbol, Function}(),          # custom_functions
        (-1.0, 1.0),                        # constant_range
        0.3,                                # constant_prob
        [:mse, :complexity],               # objectives
        Dict{Symbol, Function}(),          # custom_objectives
        nothing,                           # X
        nothing,                           # y
        Dict{Symbol, Any}(),               # extra_data
        :regression,                       # eval_mode
        200,                               # population_size
        100,                               # max_generations
        6,                                 # max_depth
        30,                                # max_nodes
        0.0,                               # validation_fraction
        nothing,                           # seed
        true                               # verbose
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Policy Problem Definition (for discrimination, sequential, etc.)
# ─────────────────────────────────────────────────────────────────────────────

"""
    PolicyProblem

A specification for symbolic policy optimization where the objective is computed
over an entire evaluation procedure (not point-by-point like regression).

Use cases:
- Discrimination problems (confirmation tracking, causation measures) → AUC-based
- Sequential problems (belief updating) → Match to normative updates
- Custom evaluation procedures

Supports constraints to ensure theoretical adequacy (e.g., directionality,
logicality for confirmation measures).

Supports ternary operators (e.g., `ifelse`) for discovering piecewise functions
like Crupi's z measure.

Created via `policy_problem()`, solved with `solve()`.
"""
mutable struct PolicyProblem
    # Grammar specification
    variables::Vector{Symbol}
    binary_operators::Vector{Function}
    unary_operators::Vector{Function}
    ternary_operators::Vector  # Can be Function or Symbol (e.g., [:ifelse])
    constant_range::Tuple{Float64, Float64}
    constant_prob::Float64
    
    # Evaluation
    evaluator::Function  # (tree, env, evaluate_fn, count_nodes_fn) -> Vector{Float64}
    n_objectives::Int
    environment::Dict{Symbol, Any}
    
    # Constraints (optional)
    constraints::Union{Nothing, Any}  # ConstraintSet or nothing
    
    # Initial population seeding (optional)
    # Each function takes variable names and returns a formula using those names
    # Example: (pH, pE, pH_E, ...) -> pH_E - pE  
    seed_formulas::Vector{Function}
    
    # Configuration
    population_size::Int
    max_generations::Int
    max_depth::Int
    max_nodes::Int
    seed::Union{Nothing, Int}
    verbose::Bool
end

"""
    policy_problem(; kwargs...) -> PolicyProblem

Create a symbolic policy optimization problem.

# Required Arguments
- `variables`: Vector of variable names (Symbols)
- `evaluator`: Function `(tree, env, evaluate_fn, count_nodes_fn) -> Vector{Float64}`
  that computes objective values for a candidate expression

# Optional Arguments
- `binary_operators`: Binary operators (default: [+, -, *, /])
- `unary_operators`: Unary operators (default: [])
- `ternary_operators`: Ternary operators for conditionals (default: [])
- `constants`: Tuple (min, max) for random constants (default: (-1.0, 1.0))
- `constant_prob`: Probability of generating constants (default: 0.3)
- `n_objectives`: Number of objectives returned by evaluator (default: 2)
- `environment`: Dict of data/parameters passed to evaluator (default: empty)
- `constraints`: ConstraintSet for theoretical requirements (default: nothing)
- `seed_formulas`: Vector of functions that build seed expressions (see example below)
- `population`, `generations`, `max_depth`, `max_nodes`, `seed`, `verbose`: Config

# Example with conditionals (for piecewise functions like z measure)
```julia
# Search for confirmation measures that can have piecewise structure
result = solve(policy_problem(
    variables = [:pH, :pE, :pH_E, :pE_H, :pE_notH, ...],
    evaluator = my_evaluator,
    # Enable conditionals to discover measures like Crupi's z
    ternary_operators = [:ifelse],
    # Optional: add step function for alternative conditional constructions
    unary_operators = [safe_step, safe_abs],
    constant_prob = 0.0,  # No arbitrary constants
    ...
))
```

The `seed_formulas` functions receive a NamedTuple of variable nodes and should
return an expression tree built using +, -, *, / operators.
"""
function policy_problem(;
    variables::Vector{Symbol},
    evaluator::Function,
    binary_operators::Vector{Function} = [+, -, *, /],
    unary_operators::Vector{Function} = Function[],
    ternary_operators::Vector = [],
    constants::Tuple{<:Real, <:Real} = (-1.0, 1.0),
    constant_prob::Float64 = 0.3,
    n_objectives::Int = 2,
    environment::Dict{Symbol, Any} = Dict{Symbol, Any}(),
    constraints = nothing,
    seed_formulas::Vector{<:Function} = Function[],
    population::Int = 200,
    generations::Int = 100,
    max_depth::Int = 6,
    max_nodes::Int = 30,
    seed::Union{Nothing, Int} = nothing,
    verbose::Bool = true
)
    PolicyProblem(
        variables,
        binary_operators,
        unary_operators,
        ternary_operators,
        (Float64(constants[1]), Float64(constants[2])),
        constant_prob,
        evaluator,
        n_objectives,
        environment,
        constraints,
        seed_formulas,
        population,
        generations,
        max_depth,
        max_nodes,
        seed,
        verbose
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Builder Functions (Fluent Interface)
# ─────────────────────────────────────────────────────────────────────────────

"""
    variables!(prob, vars...)

Set the variable names for the symbolic expressions.
"""
function variables!(prob::SymbolicProblem, vars::Symbol...)
    prob.variables = collect(vars)
    return prob
end

"""
    operators!(prob; binary=[], unary=[])

Set the operators available in symbolic expressions.
"""
function operators!(prob::SymbolicProblem; binary::Vector{Function}=Function[], unary::Vector{Function}=Function[])
    if !isempty(binary)
        prob.binary_operators = binary
    end
    if !isempty(unary)
        prob.unary_operators = unary
    end
    return prob
end

"""
    constants!(problem, range; probability=0.3)

Set the constant range and probability.
"""
function constants!(problem::SymbolicProblem, range::Tuple{<:Real, <:Real}; probability::Float64=0.3)
    problem.constant_range = (Float64(range[1]), Float64(range[2]))
    problem.constant_prob = probability
    return problem
end

"""
    add_function!(prob, name, func)

Add a custom function to the grammar.
"""
function add_function!(prob::SymbolicProblem, name::Symbol, func::Function)
    prob.custom_functions[name] = func
    return prob
end

"""
    objectives!(prob, objs...)

Set the objectives for optimization. Use :mse and :complexity for built-ins,
or symbols matching custom objectives added with `add_objective!`.
"""
function objectives!(prob::SymbolicProblem, objs::Symbol...)
    prob.objectives = collect(objs)
    return prob
end

"""
    add_objective!(prob, name, func)

Add a custom objective function. The function should take (predictions, targets) and return a scalar.
"""
function add_objective!(prob::SymbolicProblem, name::Symbol, func::Function)
    prob.custom_objectives[name] = func
    return prob
end

"""
    mode!(prob, mode::Symbol)

Set the evaluation mode:
- `:regression` (default) - Standard symbolic regression. Variables map to columns of X,
  each row is evaluated independently, predictions compared to y.
- `:aggregation` - For aggregator discovery. Variables are forecaster predictions (columns),
  the formula aggregates them into a single prediction per row.

In aggregation mode, built-in objectives like :brier and :mse compare aggregated 
predictions to y (ground truth).
"""
function mode!(prob::SymbolicProblem, mode::Symbol)
    if mode ∉ [:regression, :aggregation]
        error("Unknown mode: $mode. Use :regression or :aggregation")
    end
    prob.eval_mode = mode
    return prob
end

"""
    data!(prob; X=nothing, y=nothing, kwargs...)

Set the data for optimization.
"""
function data!(prob::SymbolicProblem; X=nothing, y=nothing, kwargs...)
    if X !== nothing
        prob.X = X isa AbstractMatrix ? X : reshape(X, :, 1)
    end
    if y !== nothing
        prob.y = vec(y)
    end
    for (k, v) in kwargs
        prob.extra_data[k] = v
    end
    return prob
end

"""
    config!(prob; kwargs...)

Set configuration options.
"""
function config!(prob::SymbolicProblem; 
                 population::Int=prob.population_size,
                 generations::Int=prob.max_generations,
                 max_depth::Int=prob.max_depth,
                 max_nodes::Int=prob.max_nodes,
                 validation::Float64=prob.validation_fraction,
                 seed::Union{Nothing,Int}=prob.seed,
                 verbose::Bool=prob.verbose)
    prob.population_size = population
    prob.max_generations = generations
    prob.max_depth = max_depth
    prob.max_nodes = max_nodes
    prob.validation_fraction = validation
    prob.seed = seed
    prob.verbose = verbose
    return prob
end

# ─────────────────────────────────────────────────────────────────────────────
# Solve Function
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve(prob::SymbolicProblem) -> SymbolicResult

Solve the symbolic optimization problem and return results.
"""
function solve(prob::SymbolicProblem)
    # Validate
    if isempty(prob.variables) && prob.X !== nothing
        # Auto-generate variable names from data dimensions
        n_vars = size(prob.X, 2)
        if prob.eval_mode == :aggregation
            # For aggregation, variables are forecasters: p1, p2, ...
            prob.variables = [Symbol("p$i") for i in 1:n_vars]
        else
            # For regression, variables are features: x1, x2, ... or just x if 1D
            prob.variables = n_vars == 1 ? [:x] : [Symbol("x$i") for i in 1:n_vars]
        end
    end
    
    if isempty(prob.variables)
        error("No variables specified. Use variables!() or provide X data.")
    end
    
    if prob.X === nothing || prob.y === nothing
        error("Data not provided. Use data!(X=..., y=...)")
    end
    
    # Build grammar
    all_unary = copy(prob.unary_operators)
    for (name, func) in prob.custom_functions
        push!(all_unary, func)
    end
    
    grammar = Grammar(
        binary_operators = prob.binary_operators,
        unary_operators = all_unary,
        variables = prob.variables,
        constant_range = prob.constant_range,
        constant_prob = prob.constant_prob
    )
    
    # Build objectives based on evaluation mode
    objective_funcs = ObjectiveFunction[]
    
    for obj in prob.objectives
        if obj == :mse
            if prob.eval_mode == :aggregation
                # Custom MSE for aggregation mode
                push!(objective_funcs, custom_objective(:mse, (ind, data) -> begin
                    X = data[:X]
                    y = data[:y]
                    var_names = data[:var_names]
                    
                    squared_errors = Float64[]
                    for i in axes(X, 1)
                        ctx = Dict(var_names[j] => X[i, j] for j in eachindex(var_names))
                        pred = evaluate(ind.tree, ctx)
                        if isfinite(pred)
                            push!(squared_errors, (pred - y[i])^2)
                        else
                            push!(squared_errors, 1.0)  # Penalty for invalid
                        end
                    end
                    return mean(squared_errors)
                end))
            else
                push!(objective_funcs, mse_objective())
            end
            
        elseif obj == :brier
            # Brier score: mean((pred - truth)^2) where truth is 0/1
            push!(objective_funcs, custom_objective(:brier, (ind, data) -> begin
                X = data[:X]
                y = data[:y]
                var_names = data[:var_names]
                
                brier_sum = 0.0
                count = 0
                for i in axes(X, 1)
                    ctx = Dict(var_names[j] => X[i, j] for j in eachindex(var_names))
                    pred = evaluate(ind.tree, ctx)
                    if isfinite(pred)
                        pred_clamped = clamp(pred, 0.0, 1.0)
                        brier_sum += (pred_clamped - y[i])^2
                        count += 1
                    else
                        brier_sum += 1.0  # Maximum penalty
                        count += 1
                    end
                end
                return count > 0 ? brier_sum / count : 1.0
            end))
            
        elseif obj == :complexity
            push!(objective_funcs, complexity_objective())
            
        elseif haskey(prob.custom_objectives, obj)
            # Wrap custom objective
            custom_func = prob.custom_objectives[obj]
            custom_obj = custom_objective(obj, (ind, data) -> begin
                X = data[:X]
                y = data[:y]
                var_names = data[:var_names]
                
                preds = Float64[]
                for i in axes(X, 1)
                    ctx = Dict(var_names[j] => X[i, j] for j in eachindex(var_names))
                    pred = evaluate(ind.tree, ctx)
                    push!(preds, isfinite(pred) ? pred : 0.0)
                end
                
                return custom_func(preds, y)
            end)
            push!(objective_funcs, custom_obj)
        else
            error("Unknown objective: $obj. Use :mse, :brier, :complexity, or add custom with add_objective!()")
        end
    end
    
    # Build config
    rng = prob.seed === nothing ? MersenneTwister() : MersenneTwister(prob.seed)
    
    config = NSGAIIConfig(
        population_size = prob.population_size,
        max_generations = prob.max_generations,
        max_depth = prob.max_depth,
        max_nodes = prob.max_nodes,
        verbose = prob.verbose
    )
    
    # Prepare data
    data_dict = Dict{Symbol, Any}(
        :X => Float64.(prob.X),
        :y => Float64.(prob.y),
        :var_names => prob.variables
    )
    
    # Add extra data
    for (k, v) in prob.extra_data
        data_dict[k] = v
    end
    
    # Run optimization
    result = optimize(grammar, objective_funcs, data_dict; config=config, rng=rng)
    
    # Wrap result
    return SymbolicResult(result, prob.variables, grammar)
end

"""
    solve(prob::PolicyProblem) -> SymbolicResult

Solve a policy optimization problem and return results.

The evaluator function is called for each candidate expression and should return
a vector of objective values to minimize.

If seed_formulas are provided, they are used to create initial population members.

If constraints are provided, they are checked and violations are penalized:
- Soft mode: Violation score is added to objectives (scaled by penalty_weight)
- Hard mode: Violating individuals receive very large objective values
"""
function solve(prob::PolicyProblem)
    # Build grammar
    grammar = Grammar(
        binary_operators = prob.binary_operators,
        unary_operators = prob.unary_operators,
        ternary_operators = prob.ternary_operators,
        variables = prob.variables,
        constant_range = prob.constant_range,
        constant_prob = prob.constant_prob
    )
    
    # Build config
    rng = prob.seed === nothing ? MersenneTwister() : MersenneTwister(prob.seed)
    
    config = NSGAIIConfig(
        population_size = prob.population_size,
        max_generations = prob.max_generations,
        max_depth = prob.max_depth,
        max_nodes = prob.max_nodes,
        verbose = prob.verbose
    )
    
    # Prepare environment with variable names
    env = copy(prob.environment)
    env[:var_names] = prob.variables
    env[:rng] = rng
    
    # Store constraints in environment for access during evaluation
    if prob.constraints !== nothing
        env[:constraints] = prob.constraints
    end
    
    # Create objective functions that use the evaluator
    # The evaluator returns a vector of objective values
    objective_funcs = ObjectiveFunction[]
    
    for i in 1:prob.n_objectives
        obj_func = custom_objective(Symbol("obj_$i"), (tree, data) -> begin
            # Call evaluator to get all objectives
            obj_values = prob.evaluator(tree, data, evaluate, count_nodes)
            
            # Apply constraint penalties if constraints are defined
            if haskey(data, :constraints) && data[:constraints] !== nothing
                cs = data[:constraints]
                
                # Check constraints
                all_satisfied, total_penalty, _ = check_constraints(cs, tree, 
                    (t, ctx) -> evaluate(t, ctx))
                
                if cs.mode == :hard && !all_satisfied
                    # Hard mode: Return very large value to effectively reject
                    return 1e10
                elseif cs.mode == :soft
                    # Soft mode: Add penalty to this objective
                    # Spread penalty across objectives
                    obj_values[i] += total_penalty / prob.n_objectives
                end
            end
            
            return obj_values[i]
        end)
        push!(objective_funcs, obj_func)
    end
    
    # Build initial population from seed_formulas if provided
    initial_pop = nothing
    if !isempty(prob.seed_formulas)
        initial_pop = Individual[]
        n_objs = prob.constraints !== nothing ? prob.n_objectives : prob.n_objectives
        for formula_fn in prob.seed_formulas
            try
                tree = build_seed_tree(formula_fn, prob.variables)
                # Create Individual with placeholder objectives (will be evaluated)
                push!(initial_pop, Individual(tree, fill(Inf, n_objs)))
            catch e
                @warn "Failed to build seed formula: $e"
            end
        end
        if prob.verbose && !isempty(initial_pop)
            println("  Seeded population with $(length(initial_pop)) formulas")
        end
    end
    
    if prob.verbose && prob.constraints !== nothing
        println("  Constraints enabled ($(prob.constraints.mode) mode, $(length(prob.constraints.constraints)) constraints)")
    end
    
    # Run optimization
    result = optimize(grammar, objective_funcs, env; 
                      config=config, rng=rng, initial_population=initial_pop)
    
    # Wrap result
    return SymbolicResult(result, prob.variables, grammar)
end

# ─────────────────────────────────────────────────────────────────────────────
# Result Wrapper
# ─────────────────────────────────────────────────────────────────────────────

"""
    SymbolicResult

Wrapper around optimization results with convenient accessors.
"""
struct SymbolicResult
    raw_result::Any
    variables::Vector{Symbol}
    grammar::Grammar
end

"""
    best(result::SymbolicResult) -> (expression::String, objectives::Vector{Float64})

Get the best solution (lowest first objective).
"""
function best(result::SymbolicResult)
    ind = get_best(result.raw_result, 1)
    expr = node_to_string(ind.tree)
    return (expression=expr, objectives=ind.objectives)
end

"""
    pareto_front(result::SymbolicResult)

Get all Pareto-optimal solutions.
"""
function pareto_front(result::SymbolicResult)
    front = get_pareto_front(result.raw_result)
    return [(expression=node_to_string(ind.tree), objectives=ind.objectives) for ind in front]
end

"""
    evaluate_best(result::SymbolicResult, X::AbstractMatrix)

Evaluate the best solution on new data.
"""
function evaluate_best(result::SymbolicResult, X::AbstractMatrix)
    ind = get_best(result.raw_result, 1)
    var_names = result.variables
    
    preds = Float64[]
    for i in axes(X, 1)
        ctx = Dict(var_names[j] => X[i, j] for j in eachindex(var_names))
        pred = evaluate(ind.tree, ctx)
        push!(preds, isfinite(pred) ? pred : 0.0)
    end
    return preds
end

# ─────────────────────────────────────────────────────────────────────────────
# High-Level Convenience Macros
# ─────────────────────────────────────────────────────────────────────────────

"""
    @symbolic_regression(X, y; kwargs...)

One-liner for basic symbolic regression.

# Example
```julia
result = @symbolic_regression(X_data, y_data, 
    operators = [+, -, *, /],
    population = 200,
    generations = 100
)
```
"""
macro symbolic_regression(X, y, kwargs...)
    # Parse keyword arguments
    kw_exprs = []
    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            push!(kw_exprs, kw)
        end
    end
    
    quote
        let prob = SymbolicProblem()
            data!(prob; X=$(esc(X)), y=$(esc(y)))
            
            # Apply kwargs
            $(if !isempty(kw_exprs)
                kw_dict = Expr(:call, :Dict, [:($(QuoteNode(kw.args[1])) => $(esc(kw.args[2]))) for kw in kw_exprs]...)
                quote
                    kw = $kw_dict
                    if haskey(kw, :operators)
                        operators!(prob; binary=kw[:operators])
                    end
                    if haskey(kw, :unary)
                        operators!(prob; unary=kw[:unary])
                    end
                    if haskey(kw, :population)
                        config!(prob; population=kw[:population])
                    end
                    if haskey(kw, :generations)
                        config!(prob; generations=kw[:generations])
                    end
                    if haskey(kw, :max_depth)
                        config!(prob; max_depth=kw[:max_depth])
                    end
                    if haskey(kw, :max_nodes)
                        config!(prob; max_nodes=kw[:max_nodes])
                    end
                    if haskey(kw, :seed)
                        config!(prob; seed=kw[:seed])
                    end
                    if haskey(kw, :verbose)
                        config!(prob; verbose=kw[:verbose])
                    end
                end
            else
                :()
            end)
            
            solve(prob)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Problem Builder Function (Alternative to macros)
# ─────────────────────────────────────────────────────────────────────────────

"""
    symbolic_problem(; kwargs...) -> SymbolicProblem

Create a symbolic optimization problem with keyword arguments.

# Example
```julia
# Standard regression
prob = symbolic_problem(
    X = my_data,
    y = my_targets,
    variables = [:x, :y, :z],
    operators = [+, -, *, /],
    population = 200,
    generations = 100
)

# Aggregator discovery
prob = symbolic_problem(
    X = forecaster_predictions,  # rows = claims, cols = forecasters
    y = ground_truth,            # 0/1 outcomes
    mode = :aggregation,
    objectives = [:brier, :complexity],
    population = 200,
    generations = 100
)
```
"""
function symbolic_problem(;
    X = nothing,
    y = nothing,
    variables::Vector{Symbol} = Symbol[],
    binary_operators::Vector{Function} = [+, -, *, /],
    unary_operators::Vector{Function} = Function[],
    constants::Tuple{<:Real, <:Real} = (-1.0, 1.0),
    constant_prob::Float64 = 0.3,
    objectives::Vector{Symbol} = [:mse, :complexity],
    mode::Symbol = :regression,
    population::Int = 200,
    generations::Int = 100,
    max_depth::Int = 6,
    max_nodes::Int = 30,
    seed::Union{Nothing, Int} = nothing,
    verbose::Bool = true,
    kwargs...  # Extra data
)
    prob = SymbolicProblem()
    
    if !isempty(variables)
        variables!(prob, variables...)
    end
    
    operators!(prob; binary=binary_operators, unary=unary_operators)
    prob.constant_range = (Float64(constants[1]), Float64(constants[2]))
    prob.constant_prob = constant_prob
    objectives!(prob, objectives...)
    mode!(prob, mode)
    
    if X !== nothing && y !== nothing
        data!(prob; X=X, y=y, kwargs...)
    end
    
    config!(prob; population=population, generations=generations, 
            max_depth=max_depth, max_nodes=max_nodes, seed=seed, verbose=verbose)
    
    return prob
end

export SymbolicProblem, SymbolicResult
export variables!, operators!, constants!, add_function!, objectives!, add_objective!, data!, config!, mode!
export solve, best, pareto_front, evaluate_best
export symbolic_problem, policy_problem

end # module DSL
