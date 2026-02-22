#=
JuMP-inspired API for Symbolic Optimization
============================================

Provides a user-friendly, declarative interface inspired by JuMP.jl.

Example:
```julia
using SymbolicOptimization

m = SymbolicModel()

@variables(m, x, y, z)
@operators(m, binary=[+, -, *, /], unary=[sin, cos])
@constants(m, -2.0..2.0, probability=0.3)

@objective(m, Min, :mse)
@objective(m, Min, :complexity)

@config(m, population=200, generations=100, seed=42)
@data(m, X=my_X, y=my_y)

optimize!(m)

best(m)
pareto_front(m)
```

For policy problems (confirmation measures, aggregators, etc.):
```julia
m = SymbolicModel()

@variables(m, pH, pE, pH_E, pE_H, pE_notH, pnotH)
@operators(m, binary=[+, -, *, /], unary=[log, safe_step])

@objective(m, Min, :auc_penalty, (tree, env) -> begin
    # custom evaluator
end)
@objective(m, Min, :complexity)

# Seed with known formulas - written in natural math syntax!
@seed(m, pH_E - pH)                    # Difference measure
@seed(m, log(pE_H / pE_notH))         # Log-likelihood ratio

@constraint(m, :neutrality, (tree, eval_fn) -> begin
    # returns (satisfied::Bool, violation_score::Float64)
end)

@data(m, trials=my_data, labels=my_labels)
@config(m, population=500, generations=150, max_depth=8)

optimize!(m)
```
=#

module API

using Random: MersenneTwister, AbstractRNG
using Statistics: mean

# Import from parent module
import ..Grammar, ..NSGAIIConfig, ..NSGAIIResult
import ..optimize, ..mse_objective, ..mae_objective, ..complexity_objective, ..depth_objective, ..custom_objective
import ..get_best, ..get_pareto_front, ..node_to_string, ..node_to_latex, ..evaluate, ..evaluate_batch
import ..Individual, ..ObjectiveFunction, ..AbstractNode, ..Variable, ..Constant, ..FunctionNode
import ..count_nodes, ..tree_depth, ..copy_tree, ..check_constraints, ..ConstraintSet, ..Constraint
import ..safe_div, ..safe_log, ..safe_exp, ..safe_sqrt, ..safe_pow
import ..safe_step, ..safe_sign, ..safe_abs, ..safe_pos, ..safe_neg
import ..safe_max, ..safe_min, ..safe_ifelse, ..soft_ifelse
import ..sigmoid, ..relu, ..softplus, ..clamp01
import ..add_objective!  # Defined in parent; API extends it for SymbolicModel

# Import best and pareto_front from DSL so we can extend them with new methods
import ..DSL: best, pareto_front

# ═══════════════════════════════════════════════════════════════════════════════
# SymbolicModel: The Central Object (like JuMP's Model)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SymbolicModel(; kwargs...)

Create a new symbolic optimization model.

A `SymbolicModel` is the central object for specifying and solving symbolic
optimization problems. It collects all problem components (grammar, objectives,
constraints, data, configuration) in one place, inspired by JuMP.jl's `Model`.

# Quick Start
```julia
m = SymbolicModel()
@variables(m, x, y)
@operators(m, binary=[+, -, *, /])
@objective(m, Min, :mse)
@objective(m, Min, :complexity)
@data(m, X=my_X, y=my_y)
optimize!(m)
best(m)
```

# Optional keyword arguments
- `seed::Int` – random seed for reproducibility
- `verbose::Bool` – print progress (default: `true`)
"""
mutable struct SymbolicModel
    # ── Grammar ──
    variable_names::Vector{Symbol}
    binary_ops::Vector{Any}       # Functions or Symbols
    unary_ops::Vector{Any}
    ternary_ops::Vector{Any}
    constant_range::Union{Nothing, Tuple{Float64, Float64}}
    constant_prob::Float64

    # ── Objectives ──
    objectives::Vector{Union{Symbol, Tuple{Symbol, Symbol, Function}}}
    # Each entry is either:
    #   :mse, :mae, :complexity, :depth  (built-in)
    #   (:name, :Min/:Max, func)         (custom)

    # ── Constraints ──
    constraints::Vector{Constraint}
    constraint_mode::Symbol       # :soft or :hard
    constraint_penalty::Float64

    # ── Seeds ──
    seed_trees::Vector{AbstractNode}

    # ── Data / Environment ──
    environment::Dict{Symbol, Any}

    # ── Configuration ──
    population_size::Int
    max_generations::Int
    tournament_size::Int
    crossover_prob::Float64
    mutation_prob::Float64
    elite_fraction::Float64
    max_depth::Int
    min_depth::Int
    max_nodes::Int
    parsimony_tolerance::Float64
    simplify_prob::Float64
    early_stop_generations::Int
    random_seed::Union{Nothing, Int}
    verbose::Bool

    # ── Results (populated after optimize!) ──
    result::Union{Nothing, NSGAIIResult}
    grammar::Union{Nothing, Grammar}
    status::Symbol   # :unsolved, :optimal, :error
end

function SymbolicModel(; seed::Union{Nothing,Int}=nothing, verbose::Bool=true)
    SymbolicModel(
        Symbol[],                         # variable_names
        Any[+, -, *, /],                  # binary_ops (sensible default)
        Any[],                            # unary_ops
        Any[],                            # ternary_ops
        (-2.0, 2.0),                      # constant_range
        0.3,                              # constant_prob
        Union{Symbol, Tuple{Symbol,Symbol,Function}}[], # objectives
        Constraint[],                     # constraints
        :soft,                            # constraint_mode
        1.0,                              # constraint_penalty
        AbstractNode[],                   # seed_trees
        Dict{Symbol, Any}(),             # environment
        200,                              # population_size
        100,                              # max_generations
        3,                                # tournament_size
        0.7,                              # crossover_prob
        0.3,                              # mutation_prob
        0.1,                              # elite_fraction
        8,                                # max_depth
        2,                                # min_depth
        50,                               # max_nodes
        0.0,                              # parsimony_tolerance
        0.1,                              # simplify_prob
        0,                                # early_stop_generations
        seed,                             # random_seed
        verbose,                          # verbose
        nothing,                          # result
        nothing,                          # grammar
        :unsolved                         # status
    )
end

function Base.show(io::IO, m::SymbolicModel)
    nvars = length(m.variable_names)
    nops = length(m.binary_ops) + length(m.unary_ops) + length(m.ternary_ops)
    nobjs = length(m.objectives)
    ncons = length(m.constraints)
    nseeds = length(m.seed_trees)
    print(io, "SymbolicModel($(m.status): $nvars vars, $nops ops, $nobjs objectives")
    ncons > 0 && print(io, ", $ncons constraints")
    nseeds > 0 && print(io, ", $nseeds seeds")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::SymbolicModel)
    println(io, "Symbolic Optimization Model")
    println(io, "═"^50)
    println(io, "Status: $(m.status)")
    println(io)

    # Variables
    if !isempty(m.variable_names)
        println(io, "Variables ($(length(m.variable_names))):")
        println(io, "  ", join(m.variable_names, ", "))
    else
        println(io, "Variables: (none)")
    end
    println(io)

    # Operators
    bin_str = isempty(m.binary_ops) ? "(none)" : join(m.binary_ops, ", ")
    println(io, "Binary operators:  $bin_str")
    if !isempty(m.unary_ops)
        println(io, "Unary operators:   ", join(m.unary_ops, ", "))
    end
    if !isempty(m.ternary_ops)
        println(io, "Ternary operators: ", join(m.ternary_ops, ", "))
    end
    println(io)

    # Constants
    if m.constant_range !== nothing
        println(io, "Constants: $(m.constant_range[1])..$(m.constant_range[2]) (prob=$(m.constant_prob))")
    else
        println(io, "Constants: disabled")
    end
    println(io)

    # Objectives
    println(io, "Objectives ($(length(m.objectives))):")
    for obj in m.objectives
        if obj isa Symbol
            println(io, "  Min $obj (built-in)")
        else
            name, sense, _ = obj
            println(io, "  $sense $name (custom)")
        end
    end
    println(io)

    # Constraints
    if !isempty(m.constraints)
        println(io, "Constraints ($(length(m.constraints)), $(m.constraint_mode) mode):")
        for c in m.constraints
            println(io, "  $(c.name): $(c.description)")
        end
        println(io)
    end

    # Seeds
    if !isempty(m.seed_trees)
        println(io, "Seed formulas ($(length(m.seed_trees))):")
        for tree in m.seed_trees
            println(io, "  $(node_to_string(tree))")
        end
        println(io)
    end

    # Config summary
    println(io, "Config: pop=$(m.population_size), gen=$(m.max_generations), depth=$(m.min_depth):$(m.max_depth), nodes≤$(m.max_nodes)")
    if m.random_seed !== nothing
        println(io, "  seed=$(m.random_seed)")
    end

    # Results
    if m.status == :optimal && m.result !== nothing
        println(io)
        println(io, "Results: $(length(m.result.pareto_front)) Pareto-optimal solutions")
        b = get_best(m.result, 1)
        println(io, "  Best: $(node_to_string(b.tree))")
        println(io, "  Objectives: $(round.(b.objectives, digits=6))")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Functional API (non-macro alternatives)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    set_variables!(m::SymbolicModel, vars::Symbol...)

Set the variable names for the model.

```julia
set_variables!(m, :x, :y, :z)
```
"""
function set_variables!(m::SymbolicModel, vars::Symbol...)
    m.variable_names = collect(vars)
    m.status = :unsolved
    return m
end

"""
    set_operators!(m::SymbolicModel; binary=nothing, unary=nothing, ternary=nothing)

Set the operators available in the grammar.

```julia
set_operators!(m, binary=[+, -, *, /], unary=[sin, cos, exp])
```
"""
function set_operators!(m::SymbolicModel; binary=nothing, unary=nothing, ternary=nothing)
    if binary !== nothing
        m.binary_ops = collect(binary)
    end
    if unary !== nothing
        m.unary_ops = collect(unary)
    end
    if ternary !== nothing
        m.ternary_ops = collect(ternary)
    end
    m.status = :unsolved
    return m
end

"""
    set_constants!(m::SymbolicModel, lo::Real, hi::Real; probability=0.3)
    set_constants!(m::SymbolicModel, ::Nothing)  # disable constants

Set the constant range and sampling probability, or disable constants.

```julia
set_constants!(m, -2.0, 2.0, probability=0.3)
set_constants!(m, nothing)  # no constants
```
"""
function set_constants!(m::SymbolicModel, lo::Real, hi::Real; probability::Real=0.3)
    m.constant_range = (Float64(lo), Float64(hi))
    m.constant_prob = Float64(probability)
    m.status = :unsolved
    return m
end

function set_constants!(m::SymbolicModel, ::Nothing)
    m.constant_range = nothing
    m.constant_prob = 0.0
    m.status = :unsolved
    return m
end

"""
    add_objective!(m::SymbolicModel, sense::Symbol, name::Symbol)
    add_objective!(m::SymbolicModel, sense::Symbol, name::Symbol, func::Function)

Add an objective to the model.

`sense` is `:Min` or `:Max`.

For built-in objectives, just pass the name:
```julia
add_objective!(m, :Min, :mse)
add_objective!(m, :Min, :complexity)
```

For custom objectives, pass a function `(tree, env) -> Float64`:
```julia
add_objective!(m, :Min, :auc_penalty, (tree, env) -> begin
    scores = [evaluate(tree, x) for x in env[:trials]]
    return 1.0 - compute_auc(scores, env[:labels])
end)
```
"""
function add_objective!(m::SymbolicModel, sense::Symbol, name::Symbol)
    @assert sense in (:Min, :Max) "Objective sense must be :Min or :Max, got :$sense"
    if name in (:mse, :mae, :complexity, :depth)
        push!(m.objectives, name)
    else
        error("Unknown built-in objective :$name. Use :mse, :mae, :complexity, or :depth, " *
              "or provide a custom function: add_objective!(m, :$sense, :$name, func)")
    end
    m.status = :unsolved
    return m
end

function add_objective!(m::SymbolicModel, sense::Symbol, name::Symbol, func::Function)
    @assert sense in (:Min, :Max) "Objective sense must be :Min or :Max, got :$sense"
    push!(m.objectives, (name, sense, func))
    m.status = :unsolved
    return m
end

"""
    add_constraint!(m::SymbolicModel, name::Symbol, test_fn::Function; description="")

Add a constraint. The `test_fn` has signature
`(tree, evaluate_fn) -> (satisfied::Bool, violation_score::Float64)`.

```julia
add_constraint!(m, :neutrality, (tree, eval_fn) -> begin
    # ... check constraint
    return (is_satisfied, violation_score)
end, description="c(H,E)=0 when P(H|E)=P(H)")
```
"""
function add_constraint!(m::SymbolicModel, name::Symbol, test_fn::Function;
                         description::String="")
    push!(m.constraints, Constraint(name, test_fn, description))
    m.status = :unsolved
    return m
end

"""
    set_constraint_mode!(m::SymbolicModel, mode::Symbol; penalty_weight=1.0)

Set how constraints are enforced: `:soft` (penalty) or `:hard` (rejection).

```julia
set_constraint_mode!(m, :hard)
set_constraint_mode!(m, :soft, penalty_weight=2.0)
```
"""
function set_constraint_mode!(m::SymbolicModel, mode::Symbol; penalty_weight::Real=1.0)
    @assert mode in (:soft, :hard) "Constraint mode must be :soft or :hard"
    m.constraint_mode = mode
    m.constraint_penalty = Float64(penalty_weight)
    m.status = :unsolved
    return m
end

"""
    add_seed!(m::SymbolicModel, tree::AbstractNode)

Add a seed expression tree to the initial population.

Prefer the `@seed` macro for natural syntax:
```julia
@seed(m, pH_E - pH)
```

For programmatic use:
```julia
add_seed!(m, FunctionNode(:-, [Variable(:pH_E), Variable(:pH)]))
```
"""
function add_seed!(m::SymbolicModel, tree::AbstractNode)
    push!(m.seed_trees, tree)
    m.status = :unsolved
    return m
end

"""
    set_data!(m::SymbolicModel; kwargs...)

Set data/environment for the optimization.

For regression problems:
```julia
set_data!(m, X=my_X, y=my_y)
```

For policy problems (custom evaluators):
```julia
set_data!(m, trials=trial_data, labels=label_data, custom_param=42)
```
"""
function set_data!(m::SymbolicModel; kwargs...)
    for (k, v) in kwargs
        m.environment[k] = v
    end
    m.status = :unsolved
    return m
end

"""
    set_config!(m::SymbolicModel; kwargs...)

Set optimization configuration.

```julia
set_config!(m,
    population = 500,
    generations = 150,
    max_depth = 8,
    max_nodes = 40,
    tournament_size = 5,
    crossover_prob = 0.7,
    mutation_prob = 0.3,
    parsimony_tolerance = 0.01,
    early_stop = 20,
    seed = 42,
    verbose = true
)
```
"""
function set_config!(m::SymbolicModel;
    population::Union{Nothing,Int} = nothing,
    generations::Union{Nothing,Int} = nothing,
    tournament_size::Union{Nothing,Int} = nothing,
    crossover_prob::Union{Nothing,Real} = nothing,
    mutation_prob::Union{Nothing,Real} = nothing,
    elite_fraction::Union{Nothing,Real} = nothing,
    max_depth::Union{Nothing,Int} = nothing,
    min_depth::Union{Nothing,Int} = nothing,
    max_nodes::Union{Nothing,Int} = nothing,
    parsimony_tolerance::Union{Nothing,Real} = nothing,
    simplify_prob::Union{Nothing,Real} = nothing,
    early_stop::Union{Nothing,Int} = nothing,
    seed::Union{Nothing,Int} = nothing,
    verbose::Union{Nothing,Bool} = nothing,
)
    population !== nothing         && (m.population_size = population)
    generations !== nothing        && (m.max_generations = generations)
    tournament_size !== nothing    && (m.tournament_size = tournament_size)
    crossover_prob !== nothing     && (m.crossover_prob = Float64(crossover_prob))
    mutation_prob !== nothing      && (m.mutation_prob = Float64(mutation_prob))
    elite_fraction !== nothing     && (m.elite_fraction = Float64(elite_fraction))
    max_depth !== nothing          && (m.max_depth = max_depth)
    min_depth !== nothing          && (m.min_depth = min_depth)
    max_nodes !== nothing          && (m.max_nodes = max_nodes)
    parsimony_tolerance !== nothing && (m.parsimony_tolerance = Float64(parsimony_tolerance))
    simplify_prob !== nothing      && (m.simplify_prob = Float64(simplify_prob))
    early_stop !== nothing         && (m.early_stop_generations = early_stop)
    seed !== nothing               && (m.random_seed = seed)
    verbose !== nothing            && (m.verbose = verbose)
    m.status = :unsolved
    return m
end

# ═══════════════════════════════════════════════════════════════════════════════
# Expression Tree Builder: Convert Julia AST → Expression Tree
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the core of the @seed macro: it takes a Julia expression like
# `pH_E - pH` or `log(pE_H / pE_notH)` and converts it into an AbstractNode tree.

"""
    expr_to_tree(expr, var_names::Set{Symbol}) -> AbstractNode

Convert a Julia expression AST into an expression tree.

Recognizes:
- Symbols that match variable names → Variable nodes
- Numeric literals → Constant nodes
- Function calls → FunctionNode with the function name as symbol
- Infix operations (+, -, *, /) → FunctionNode
- Unary minus (-x) → FunctionNode(:*, [Constant(-1.0), x])
"""
function expr_to_tree(expr, var_names::Set{Symbol})
    if expr isa Symbol
        if expr in var_names
            return Variable(expr)
        else
            # Could be a function name used as a variable – treat as variable anyway
            # to allow flexible experimentation
            return Variable(expr)
        end
    elseif expr isa Real
        return Constant(Float64(expr))
    elseif expr isa Expr
        if expr.head == :call
            func_name = expr.args[1]
            args = expr.args[2:end]

            # Get the symbol for the function
            fname = if func_name isa Symbol
                func_name
            elseif func_name isa Function
                nameof(func_name)
            else
                Symbol(string(func_name))
            end

            # Handle unary minus: -x  →  (-1.0) * x
            if fname == :- && length(args) == 1
                child = expr_to_tree(args[1], var_names)
                return FunctionNode(:*, [Constant(-1.0), child])
            end

            # Handle unary plus: +x  →  x
            if fname == :+ && length(args) == 1
                return expr_to_tree(args[1], var_names)
            end

            # Recursively convert arguments
            child_trees = [expr_to_tree(a, var_names) for a in args]
            return FunctionNode(fname, child_trees)
        elseif expr.head == :block
            # Handle begin...end blocks: use the last expression
            # Filter out LineNumberNode
            exprs = filter(e -> !(e isa LineNumberNode), expr.args)
            if length(exprs) == 1
                return expr_to_tree(exprs[1], var_names)
            else
                error("Seed expression block must contain exactly one expression, got $(length(exprs))")
            end
        else
            error("Unsupported expression type in @seed: $(expr.head)")
        end
    else
        error("Cannot convert to expression tree: $expr (type: $(typeof(expr)))")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Macros: The JuMP-like Interface
# ═══════════════════════════════════════════════════════════════════════════════

"""
    @variables(m, v1, v2, ...)

Declare the variables available in symbolic expressions.
No need to quote with `:` – bare symbols work.

```julia
m = SymbolicModel()
@variables(m, x, y, z)
@variables(m, pH, pE, pH_E, pE_H, pE_notH, pnotH)
```
"""
macro variables(model, vars...)
    var_syms = [QuoteNode(v) for v in vars]
    quote
        set_variables!($(esc(model)), $(var_syms...))
    end
end

"""
    @operators(m, binary=[...], unary=[...], ternary=[...])

Set the operators for the grammar.

```julia
@operators(m, binary=[+, -, *, /], unary=[sin, cos, exp])
@operators(m, binary=[+, -, *, /], unary=[log, safe_step], ternary=[safe_ifelse])
```
"""
macro operators(model, kwargs...)
    kw_exprs = Expr[]
    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            push!(kw_exprs, Expr(:kw, kw.args[1], esc(kw.args[2])))
        else
            error("@operators expects keyword arguments, e.g., @operators(m, binary=[+,-,*,/])")
        end
    end
    quote
        set_operators!($(esc(model)); $(kw_exprs...))
    end
end

"""
    @constants(m, lo..hi)
    @constants(m, lo..hi, probability=0.3)
    @constants(m, nothing)

Set the constant range, or disable constants.

```julia
@constants(m, -2.0..2.0)
@constants(m, -2.0..2.0, probability=0.5)
@constants(m, nothing)  # no constants
```
"""
macro constants(model, range_expr, kwargs...)
    if range_expr === :nothing || range_expr === nothing
        return quote
            set_constants!($(esc(model)), nothing)
        end
    end

    # Parse lo..hi range expression
    if range_expr isa Expr && range_expr.head == :call && range_expr.args[1] == :(..)
        lo = range_expr.args[2]
        hi = range_expr.args[3]
    elseif range_expr isa Expr && range_expr.head == :call && range_expr.args[1] == :(:)
        lo = range_expr.args[2]
        hi = range_expr.args[3]
    else
        error("@constants expects a range like -2.0..2.0 or -2.0:2.0, got: $range_expr")
    end

    # Parse keyword arguments
    kw_exprs = Expr[]
    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            push!(kw_exprs, Expr(:kw, kw.args[1], esc(kw.args[2])))
        end
    end

    quote
        set_constants!($(esc(model)), $(esc(lo)), $(esc(hi)); $(kw_exprs...))
    end
end

"""
    @objective(m, Min, :name)
    @objective(m, Min, :name, func)
    @objective(m, Max, :name, func)

Add an optimization objective. Use `Min` or `Max` for the sense.

Built-in objectives: `:mse`, `:mae`, `:complexity`, `:depth`

```julia
# Built-in
@objective(m, Min, :mse)
@objective(m, Min, :complexity)

# Custom: func receives (tree::AbstractNode, env::Dict{Symbol,Any})
@objective(m, Min, :auc_penalty, (tree, env) -> begin
    scores = [evaluate(tree, x) for x in env[:trials]]
    return 1.0 - compute_auc(scores, env[:labels])
end)

# Maximize example
@objective(m, Max, :accuracy, (tree, env) -> begin
    # ... compute accuracy
end)
```
"""
macro objective(model, sense, name, func=nothing)
    sense_sym = QuoteNode(sense)
    name_sym = if name isa QuoteNode
        name
    elseif name isa Symbol
        QuoteNode(name)
    else
        esc(name)
    end

    if func === nothing
        quote
            add_objective!($(esc(model)), $sense_sym, $name_sym)
        end
    else
        quote
            add_objective!($(esc(model)), $sense_sym, $name_sym, $(esc(func)))
        end
    end
end

"""
    @constraint(m, :name, test_fn; description="...")

Add a constraint. The `test_fn` has signature
`(tree, evaluate_fn) -> (satisfied::Bool, violation_score::Float64)`.

```julia
@constraint(m, :neutrality, (tree, eval_fn) -> begin
    violations = 0
    for ctx in neutral_cases
        val = eval_fn(tree, ctx)
        violations += abs(val) > 0.05 ? 1 : 0
    end
    score = violations / length(neutral_cases)
    return (score < 0.05, score)
end, description="c(H,E)=0 when P(H|E)=P(H)")
```
"""
macro constraint(model, name, func, kwargs...)
    name_sym = if name isa QuoteNode
        name
    elseif name isa Symbol
        QuoteNode(name)
    else
        esc(name)
    end

    kw_exprs = Expr[]
    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            push!(kw_exprs, Expr(:kw, kw.args[1], esc(kw.args[2])))
        end
    end

    quote
        add_constraint!($(esc(model)), $name_sym, $(esc(func)); $(kw_exprs...))
    end
end

"""
    @seed(m, expression)

Add a seed formula to the initial population, written in natural math syntax.
Variable names must match those declared with `@variables`.

This is the key ergonomic feature: instead of manually building trees, write
formulas as you would on paper.

```julia
@variables(m, pH, pE, pH_E, pE_H, pE_notH, pnotH)

# Difference measure: d = P(H|E) - P(H)
@seed(m, pH_E - pH)

# Log-likelihood ratio: l = log(P(E|H) / P(E|¬H))
@seed(m, log(pE_H / pE_notH))

# Christensen's s: P(H|E) - P(H|¬E)
@seed(m, pH_E - pH_notE)

# Compound expression
@seed(m, (pH_E - pH) / (1.0 - pH))

# With unary operators
@seed(m, sin(x) + cos(y))
```
"""
macro seed(model, expr)
    # We capture the expression AST and convert it to tree-building code at runtime
    # This is necessary because we need the model's variable names at runtime
    quoted_expr = QuoteNode(expr)
    quote
        let m = $(esc(model))
            var_set = Set(m.variable_names)
            tree = expr_to_tree($quoted_expr, var_set)
            add_seed!(m, tree)
        end
    end
end

"""
    @data(m, key1=val1, key2=val2, ...)

Set data/environment for the optimization.

For regression problems, use `X` and `y`:
```julia
@data(m, X=my_features, y=my_targets)
```

For policy problems, pass arbitrary named data:
```julia
@data(m, trials=trial_data, labels=label_data, n_worlds=20)
```
"""
macro data(model, kwargs...)
    kw_exprs = Expr[]
    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            push!(kw_exprs, Expr(:kw, kw.args[1], esc(kw.args[2])))
        else
            error("@data expects keyword arguments, e.g., @data(m, X=data, y=targets)")
        end
    end
    quote
        set_data!($(esc(model)); $(kw_exprs...))
    end
end

"""
    @config(m, key1=val1, key2=val2, ...)

Set optimization configuration.

```julia
@config(m,
    population = 500,
    generations = 150,
    max_depth = 8,
    max_nodes = 40,
    tournament_size = 5,
    parsimony_tolerance = 0.01,
    early_stop = 20,
    seed = 42,
    verbose = true
)
```
"""
macro config(model, kwargs...)
    kw_exprs = Expr[]
    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            push!(kw_exprs, Expr(:kw, kw.args[1], esc(kw.args[2])))
        else
            error("@config expects keyword arguments")
        end
    end
    quote
        set_config!($(esc(model)); $(kw_exprs...))
    end
end

"""
    @grammar(m) do ... end
    @grammar(m, begin ... end)

Define the entire grammar in one block.

```julia
@grammar(m, begin
    variables: x, y, z
    binary:    +, -, *, /
    unary:     sin, cos, exp
    constants: -2.0..2.0, probability=0.3
end)
```
"""
macro grammar(model, block)
    if !(block isa Expr && block.head == :block)
        error("@grammar expects a begin...end block")
    end

    stmts = Any[]
    m = esc(model)

    for expr in block.args
        expr isa LineNumberNode && continue

        if expr isa Expr && expr.head == :call && expr.args[1] == :(:)
            label = expr.args[2]
            rest = expr.args[3:end]

            if label == :variables
                var_syms = [QuoteNode(v) for v in rest]
                push!(stmts, :(set_variables!($m, $(var_syms...))))
            elseif label == :binary
                push!(stmts, :(set_operators!($m; binary=$(esc(Expr(:vect, rest...))))))
            elseif label == :unary
                push!(stmts, :(set_operators!($m; unary=$(esc(Expr(:vect, rest...))))))
            elseif label == :ternary
                push!(stmts, :(set_operators!($m; ternary=$(esc(Expr(:vect, rest...))))))
            elseif label == :constants
                # Handle  `constants: -2.0..2.0`  or  `constants: -2.0..2.0, probability=0.3`
                # The range expression parses as Expr(:call, :.., lo, hi)
                range_expr = nothing
                prob_expr = nothing
                for arg in rest
                    if arg isa Expr && arg.head == :call && arg.args[1] == :(..)
                        range_expr = arg
                    elseif arg isa Expr && arg.head == :(=) && arg.args[1] == :probability
                        prob_expr = arg.args[2]
                    elseif arg isa Expr && arg.head == :kw && arg.args[1] == :probability
                        prob_expr = arg.args[2]
                    end
                end
                if range_expr !== nothing
                    lo = esc(range_expr.args[2])
                    hi = esc(range_expr.args[3])
                    if prob_expr !== nothing
                        push!(stmts, :(set_constants!($m, $lo, $hi; probability=$(esc(prob_expr)))))
                    else
                        push!(stmts, :(set_constants!($m, $lo, $hi)))
                    end
                else
                    @warn "@grammar: could not parse constants specification"
                end
            else
                @warn "Unknown @grammar label: $label"
            end
        elseif expr isa Expr && expr.head == :call
            # Handle  `constants: -2.0..2.0`  style
            # This might appear as a different parse – handled above
        end
    end

    quote
        $(stmts...)
        $m
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# optimize! — The Main Solve Function
# ═══════════════════════════════════════════════════════════════════════════════

"""
    optimize!(m::SymbolicModel) -> SymbolicModel

Run the symbolic optimization. Populates `m` with results accessible via
`best(m)`, `pareto_front(m)`, etc.

```julia
m = SymbolicModel()
# ... set up model ...
optimize!(m)

# Access results
best(m)                    # best solution
pareto_front(m)            # all Pareto-optimal solutions
best(m, objective=2)       # best for 2nd objective
objective_value(m)         # objective values of the best
expression_string(m)       # string form of best expression
expression_latex(m)        # LaTeX form
```
"""
function optimize!(m::SymbolicModel)
    # ── Validation ──
    if isempty(m.variable_names)
        # Try to auto-infer from data
        if haskey(m.environment, :X)
            X = m.environment[:X]
            ncols = size(X, 2)
            m.variable_names = ncols == 1 ? [:x] : [Symbol("x$i") for i in 1:ncols]
        else
            error("No variables specified. Use @variables(m, ...) or set_variables!(m, ...)")
        end
    end

    if isempty(m.objectives)
        error("No objectives specified. Use @objective(m, ...) or add_objective!(m, ...)")
    end

    # ── Build Grammar ──
    grammar_kwargs = Dict{Symbol, Any}()
    grammar_kwargs[:variables] = m.variable_names
    grammar_kwargs[:binary_operators] = _resolve_ops(m.binary_ops)
    grammar_kwargs[:constant_prob] = m.constant_prob

    if !isempty(m.unary_ops)
        grammar_kwargs[:unary_operators] = _resolve_ops(m.unary_ops)
    end
    if !isempty(m.ternary_ops)
        grammar_kwargs[:ternary_operators] = _resolve_ops(m.ternary_ops)
    end
    if m.constant_range !== nothing
        grammar_kwargs[:constant_range] = m.constant_range
    else
        grammar_kwargs[:constant_range] = nothing
    end

    grammar = Grammar(; grammar_kwargs...)
    m.grammar = grammar

    # ── Build Objectives ──
    objective_funcs = _build_objectives(m)

    # ── Build Environment ──
    env = copy(m.environment)
    env[:var_names] = m.variable_names

    # ── Build Constraints ──
    if !isempty(m.constraints)
        cs = ConstraintSet(m.constraints; mode=m.constraint_mode, penalty_weight=m.constraint_penalty)
        env[:constraints] = cs
    end

    # ── RNG ──
    rng = m.random_seed === nothing ? MersenneTwister() : MersenneTwister(m.random_seed)
    env[:rng] = rng

    # ── Config ──
    config = NSGAIIConfig(
        population_size = m.population_size,
        max_generations = m.max_generations,
        tournament_size = m.tournament_size,
        crossover_prob = m.crossover_prob,
        mutation_prob = m.mutation_prob,
        elite_fraction = m.elite_fraction,
        max_depth = m.max_depth,
        min_depth = m.min_depth,
        max_nodes = m.max_nodes,
        parsimony_tolerance = m.parsimony_tolerance,
        simplify_prob = m.simplify_prob,
        verbose = m.verbose,
        early_stop_generations = m.early_stop_generations,
    )

    # ── Initial Population from Seeds ──
    initial_pop = nothing
    if !isempty(m.seed_trees)
        initial_pop = Individual[]
        for tree in m.seed_trees
            try
                push!(initial_pop, Individual(copy_tree(tree)))  # empty objectives → will be evaluated
            catch e
                m.verbose && @warn "Failed to create seed individual: $e"
            end
        end
        m.verbose && !isempty(initial_pop) && println("  Seeded population with $(length(initial_pop)) formulas")
    end

    # ── Run Optimization ──
    try
        result = optimize(grammar, objective_funcs, env;
                         config=config, rng=rng, initial_population=initial_pop)
        m.result = result
        m.status = :optimal
    catch e
        m.status = :error
        rethrow(e)
    end

    return m
end

# ── Helper: Resolve operator specs (Function or Symbol → Function) ──
function _resolve_ops(ops)
    resolved = Any[]
    for op in ops
        if op isa Function
            push!(resolved, op)
        elseif op isa Symbol
            # Try to find as a safe implementation
            push!(resolved, op)
        else
            push!(resolved, op)
        end
    end
    return resolved
end

# ── Helper: Build ObjectiveFunction vector from model ──
function _build_objectives(m::SymbolicModel)
    objective_funcs = ObjectiveFunction[]
    has_regression_data = haskey(m.environment, :X) && haskey(m.environment, :y)

    for obj in m.objectives
        if obj isa Symbol
            # Built-in objectives
            if obj == :mse
                if has_regression_data
                    push!(objective_funcs, mse_objective())
                else
                    push!(objective_funcs, _policy_mse_objective(m.variable_names))
                end
            elseif obj == :mae
                push!(objective_funcs, mae_objective())
            elseif obj == :complexity
                push!(objective_funcs, complexity_objective())
            elseif obj == :depth
                push!(objective_funcs, depth_objective())
            else
                error("Unknown built-in objective: :$obj")
            end
        else
            # Custom objective: (name, sense, func)
            name, sense, func = obj
            minimize = (sense == :Min)
            push!(objective_funcs, custom_objective(name, func; minimize=minimize))
        end
    end

    return objective_funcs
end

# ── Helper: MSE for policy-style problems ──
function _policy_mse_objective(var_names::Vector{Symbol})
    custom_objective(:mse, (tree, data) -> begin
        X = data[:X]
        y = data[:y]
        vnames = get(data, :var_names, var_names)
        squared_errors = Float64[]
        for i in axes(X, 1)
            ctx = Dict(vnames[j] => X[i, j] for j in eachindex(vnames))
            pred = evaluate(tree, ctx)
            if isfinite(pred)
                push!(squared_errors, (pred - y[i])^2)
            else
                push!(squared_errors, 1.0)
            end
        end
        return mean(squared_errors)
    end)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Result Accessors
# ═══════════════════════════════════════════════════════════════════════════════

"""
    best(m::SymbolicModel; objective=1) -> NamedTuple

Get the best solution for a given objective (default: first).

Returns a NamedTuple with:
- `tree`: the AbstractNode expression tree
- `expression`: string representation
- `latex`: LaTeX representation
- `objectives`: vector of objective values
- `rank`: Pareto rank

```julia
b = best(m)
b.expression   # "pH_E - pH"
b.objectives   # [0.123, 5.0]
b.tree         # AbstractNode
```
"""
function best(m::SymbolicModel; objective::Int=1)
    _check_solved(m)
    ind = get_best(m.result, objective)
    return (
        tree = ind.tree,
        expression = node_to_string(ind.tree),
        latex = node_to_latex(ind.tree),
        objectives = ind.objectives,
        rank = ind.rank,
    )
end

"""
    pareto_front(m::SymbolicModel) -> Vector{NamedTuple}

Get all Pareto-optimal solutions.

```julia
for sol in pareto_front(m)
    println("\$(sol.expression)  objectives=\$(sol.objectives)")
end
```
"""
function pareto_front(m::SymbolicModel)
    _check_solved(m)
    front = get_pareto_front(m.result)
    return [
        (
            tree = ind.tree,
            expression = node_to_string(ind.tree),
            latex = node_to_latex(ind.tree),
            objectives = ind.objectives,
            rank = ind.rank,
        )
        for ind in front
    ]
end

"""
    objective_value(m::SymbolicModel; objective=1) -> Float64

Get the best objective value for a specific objective.
"""
function objective_value(m::SymbolicModel; objective::Int=1)
    _check_solved(m)
    return get_best(m.result, objective).objectives[objective]
end

"""
    expression_string(m::SymbolicModel; objective=1) -> String

Get the string representation of the best expression.
"""
function expression_string(m::SymbolicModel; objective::Int=1)
    _check_solved(m)
    return node_to_string(get_best(m.result, objective).tree)
end

"""
    expression_latex(m::SymbolicModel; objective=1) -> String

Get the LaTeX representation of the best expression.
"""
function expression_latex(m::SymbolicModel; objective::Int=1)
    _check_solved(m)
    return node_to_latex(get_best(m.result, objective).tree)
end

"""
    predict(m::SymbolicModel, X::AbstractMatrix; objective=1) -> Vector{Float64}

Evaluate the best expression on new data.

```julia
X_test = [1.0 2.0; 3.0 4.0]
preds = predict(m, X_test)
```
"""
function predict(m::SymbolicModel, X::AbstractMatrix; objective::Int=1)
    _check_solved(m)
    ind = get_best(m.result, objective)
    var_names = m.variable_names
    return evaluate_batch(ind.tree, X, var_names)
end

"""
    history(m::SymbolicModel) -> Vector{Dict}

Get the per-generation optimization history.
"""
function history(m::SymbolicModel)
    _check_solved(m)
    return m.result.history
end

"""
    raw_result(m::SymbolicModel) -> NSGAIIResult

Get the underlying NSGAIIResult for advanced use.
"""
function raw_result(m::SymbolicModel)
    _check_solved(m)
    return m.result
end

function _check_solved(m::SymbolicModel)
    if m.status == :unsolved
        error("Model has not been optimized yet. Call optimize!(m) first.")
    elseif m.status == :error
        error("Last optimization failed. Check error and try again.")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# @problem — All-in-One Block Macro (the "JuMP @model" equivalent)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    @problem(block) -> SymbolicModel

Define and (optionally) solve a symbolic optimization problem in a single block.
Returns a configured `SymbolicModel`.

```julia
m = @problem begin
    @variables pH, pE, pH_E, pE_H, pE_notH, pnotH

    @operators binary=[+, -, *, /] unary=[log]
    @constants nothing

    @objective Min :auc_penalty (tree, env) -> begin
        # custom evaluator
    end
    @objective Min :complexity

    @seed pH_E - pH
    @seed log(pE_H / pE_notH)

    @data trials=my_data labels=my_labels

    @config population=500 generations=150 seed=42
end

optimize!(m)
best(m)
```
"""
macro problem(block)
    if !(block isa Expr && block.head == :block)
        error("@problem expects a begin...end block")
    end

    stmts = Any[:(m = SymbolicModel())]

    for expr in block.args
        expr isa LineNumberNode && continue

        if expr isa Expr && expr.head == :macrocall
            macroname = expr.args[1]
            # Strip the @-prefix to get the macro symbol
            mname = string(macroname)

            if mname in ("@variables", "@__dot__variables")
                # @variables x, y, z → set_variables!(m, :x, :y, :z)
                var_args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
                var_syms = [QuoteNode(v) for v in var_args]
                push!(stmts, :(set_variables!(m, $(var_syms...))))
            elseif mname in ("@operators", "@__dot__operators")
                args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
                kw_exprs = Expr[]
                for kw in args
                    if kw isa Expr && kw.head == :(=)
                        push!(kw_exprs, Expr(:kw, kw.args[1], esc(kw.args[2])))
                    end
                end
                push!(stmts, :(set_operators!(m; $(kw_exprs...))))
            elseif mname in ("@constants", "@__dot__constants")
                args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
                if length(args) == 1 && args[1] === :nothing
                    push!(stmts, :(set_constants!(m, nothing)))
                else
                    range_expr = args[1]
                    if range_expr isa Expr && range_expr.head == :call && range_expr.args[1] == :(..)
                        lo, hi = range_expr.args[2], range_expr.args[3]
                        kw_exprs = Expr[]
                        for kw in args[2:end]
                            if kw isa Expr && kw.head == :(=)
                                push!(kw_exprs, Expr(:kw, kw.args[1], esc(kw.args[2])))
                            end
                        end
                        push!(stmts, :(set_constants!(m, $(esc(lo)), $(esc(hi)); $(kw_exprs...))))
                    end
                end
            elseif mname in ("@objective", "@__dot__objective")
                args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
                sense = QuoteNode(args[1])
                name = if args[2] isa QuoteNode; args[2]
                       elseif args[2] isa Symbol; QuoteNode(args[2])
                       else; esc(args[2])
                       end
                if length(args) >= 3
                    push!(stmts, :(add_objective!(m, $sense, $name, $(esc(args[3])))))
                else
                    push!(stmts, :(add_objective!(m, $sense, $name)))
                end
            elseif mname in ("@seed", "@__dot__seed")
                args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
                seed_expr = QuoteNode(args[1])
                push!(stmts, :(add_seed!(m, expr_to_tree($seed_expr, Set(m.variable_names)))))
            elseif mname in ("@data", "@__dot__data")
                args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
                kw_exprs = Expr[]
                for kw in args
                    if kw isa Expr && kw.head == :(=)
                        push!(kw_exprs, Expr(:kw, kw.args[1], esc(kw.args[2])))
                    end
                end
                push!(stmts, :(set_data!(m; $(kw_exprs...))))
            elseif mname in ("@config", "@__dot__config")
                args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
                kw_exprs = Expr[]
                for kw in args
                    if kw isa Expr && kw.head == :(=)
                        push!(kw_exprs, Expr(:kw, kw.args[1], esc(kw.args[2])))
                    end
                end
                push!(stmts, :(set_config!(m; $(kw_exprs...))))
            elseif mname in ("@constraint", "@__dot__constraint")
                args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
                name_sym = if args[1] isa QuoteNode; args[1]
                           elseif args[1] isa Symbol; QuoteNode(args[1])
                           else; esc(args[1])
                           end
                push!(stmts, :(add_constraint!(m, $name_sym, $(esc(args[2])))))
            end
        end
    end

    push!(stmts, :m)

    return Expr(:block, stmts...)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════════

export SymbolicModel
export set_variables!, set_operators!, set_constants!, add_constraint!
export set_constraint_mode!, add_seed!, set_data!, set_config!
export optimize!
# Note: add_objective! is defined in parent module and extended here — not re-exported
# Note: best() and pareto_front() are extended from DSL module, not re-exported
export objective_value, expression_string, expression_latex
export predict, history, raw_result
export @variables, @operators, @constants, @objective, @constraint, @seed, @data, @config, @grammar
export @problem
export expr_to_tree

end # module API
