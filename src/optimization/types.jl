# ═══════════════════════════════════════════════════════════════════════════════
# NSGA-II Types and Structures
# ═══════════════════════════════════════════════════════════════════════════════

using Random

# ───────────────────────────────────────────────────────────────────────────────
# Individual
# ───────────────────────────────────────────────────────────────────────────────

"""
    Individual

An individual in the evolutionary population, wrapping an expression tree
with its fitness values and NSGA-II ranking information.

# Fields
- `tree::AbstractNode`: The expression tree
- `objectives::Vector{Float64}`: Fitness values for each objective
- `rank::Int`: Pareto rank (1 = first front, 2 = second front, etc.)
- `crowding_distance::Float64`: Crowding distance for diversity
- `data::Dict{Symbol,Any}`: Optional metadata storage
"""
mutable struct Individual
    tree::AbstractNode
    objectives::Vector{Float64}
    rank::Int
    crowding_distance::Float64
    data::Dict{Symbol,Any}
end

function Individual(tree::AbstractNode)
    Individual(tree, Float64[], 0, 0.0, Dict{Symbol,Any}())
end

function Individual(tree::AbstractNode, objectives::Vector{Float64})
    Individual(tree, objectives, 0, 0.0, Dict{Symbol,Any}())
end

# Copy constructor
function Base.copy(ind::Individual)
    Individual(
        copy_tree(ind.tree),
        copy(ind.objectives),
        ind.rank,
        ind.crowding_distance,
        copy(ind.data)
    )
end

function Base.show(io::IO, ind::Individual)
    obj_str = isempty(ind.objectives) ? "[]" : "[$(join(round.(ind.objectives, digits=4), ", "))]"
    print(io, "Individual(rank=$(ind.rank), obj=$obj_str, tree=$(node_to_string(ind.tree)))")
end

# ───────────────────────────────────────────────────────────────────────────────
# Objective Function
# ───────────────────────────────────────────────────────────────────────────────

"""
    ObjectiveFunction

Defines an objective to optimize.

# Fields
- `name::Symbol`: Name of the objective
- `func::Function`: Function (tree, data) -> Float64
- `minimize::Bool`: If true, lower is better; if false, higher is better
- `weight::Float64`: Optional weight for weighted-sum methods
"""
struct ObjectiveFunction
    name::Symbol
    func::Function
    minimize::Bool
    weight::Float64
end

function ObjectiveFunction(name::Symbol, func::Function; minimize::Bool=true, weight::Float64=1.0)
    ObjectiveFunction(name, func, minimize, weight)
end

function Base.show(io::IO, obj::ObjectiveFunction)
    dir = obj.minimize ? "min" : "max"
    print(io, "Objective(:$(obj.name), $dir)")
end

# ───────────────────────────────────────────────────────────────────────────────
# Common Objectives
# ───────────────────────────────────────────────────────────────────────────────

"""
    mse_objective(name=:mse) -> ObjectiveFunction

Create a Mean Squared Error objective.
Expects `data` to contain `:X` (input matrix) and `:y` (target vector).
"""
function mse_objective(; name::Symbol=:mse)
    ObjectiveFunction(name, _compute_mse; minimize=true)
end

function _compute_mse(tree::AbstractNode, data::Dict)
    X = data[:X]
    y = data[:y]
    var_names = get(data, :var_names, [:x])
    grammar = get(data, :grammar, nothing)
    
    predictions = grammar === nothing ? 
        evaluate_batch(tree, X, var_names) :
        evaluate_batch(tree, grammar, X, var_names)
    
    # Handle NaN predictions
    valid = .!isnan.(predictions)
    if sum(valid) == 0
        return Inf
    end
    
    mse = sum((predictions[valid] .- y[valid]).^2) / sum(valid)
    return isfinite(mse) ? mse : Inf
end

"""
    mae_objective(name=:mae) -> ObjectiveFunction

Create a Mean Absolute Error objective.
"""
function mae_objective(; name::Symbol=:mae)
    ObjectiveFunction(name, _compute_mae; minimize=true)
end

function _compute_mae(tree::AbstractNode, data::Dict)
    X = data[:X]
    y = data[:y]
    var_names = get(data, :var_names, [:x])
    grammar = get(data, :grammar, nothing)
    
    predictions = grammar === nothing ?
        evaluate_batch(tree, X, var_names) :
        evaluate_batch(tree, grammar, X, var_names)
    
    valid = .!isnan.(predictions)
    if sum(valid) == 0
        return Inf
    end
    
    mae = sum(abs.(predictions[valid] .- y[valid])) / sum(valid)
    return isfinite(mae) ? mae : Inf
end

"""
    complexity_objective(name=:complexity) -> ObjectiveFunction

Create a tree complexity objective (number of nodes).
"""
function complexity_objective(; name::Symbol=:complexity)
    ObjectiveFunction(name, (tree, data) -> Float64(count_nodes(tree)); minimize=true)
end

"""
    depth_objective(name=:depth) -> ObjectiveFunction

Create a tree depth objective.
"""
function depth_objective(; name::Symbol=:depth)
    ObjectiveFunction(name, (tree, data) -> Float64(tree_depth(tree)); minimize=true)
end

"""
    custom_objective(name, func; minimize=true, weight=1.0) -> ObjectiveFunction

Create a custom objective function.

# Arguments
- `name`: Name of the objective
- `func`: Function with signature `(tree::AbstractNode, data::Dict) -> Float64`
- `minimize`: If true, lower values are better
- `weight`: Weight for weighted-sum methods
"""
function custom_objective(name::Symbol, func::Function; minimize::Bool=true, weight::Float64=1.0)
    ObjectiveFunction(name, func; minimize=minimize, weight=weight)
end

# ───────────────────────────────────────────────────────────────────────────────
# NSGA-II Configuration
# ───────────────────────────────────────────────────────────────────────────────

"""
    NSGAIIConfig

Configuration for the NSGA-II algorithm.

# Fields
- `population_size::Int`: Number of individuals in the population
- `max_generations::Int`: Maximum number of generations
- `tournament_size::Int`: Tournament selection size
- `crossover_prob::Float64`: Probability of crossover vs mutation
- `mutation_prob::Float64`: Probability of mutation (when not doing crossover)
- `elite_fraction::Float64`: Fraction of population to preserve as elite
- `max_depth::Int`: Maximum tree depth
- `min_depth::Int`: Minimum tree depth for generation
- `max_nodes::Int`: Maximum number of nodes per tree (0 = unlimited)
- `parsimony_tolerance::Float64`: If two solutions' primary objectives differ by less than this fraction, prefer the simpler one (0 = disabled)
- `simplify_prob::Float64`: Probability of simplifying offspring
- `verbose::Bool`: Print progress information
- `early_stop_generations::Int`: Stop if no improvement for this many generations (0 = disabled)
"""
@kwdef struct NSGAIIConfig
    population_size::Int = 100
    max_generations::Int = 50
    tournament_size::Int = 3
    crossover_prob::Float64 = 0.7
    mutation_prob::Float64 = 0.3
    elite_fraction::Float64 = 0.1
    max_depth::Int = 8
    min_depth::Int = 2
    max_nodes::Int = 50              # Maximum nodes per tree (0 = unlimited)
    parsimony_tolerance::Float64 = 0.0  # Tolerance for preferring simpler solutions (e.g., 0.01 = 1%)
    simplify_prob::Float64 = 0.1
    verbose::Bool = true
    early_stop_generations::Int = 0
end

function Base.show(io::IO, config::NSGAIIConfig)
    nodes_str = config.max_nodes > 0 ? ", max_nodes=$(config.max_nodes)" : ""
    parsimony_str = config.parsimony_tolerance > 0 ? ", parsimony=$(config.parsimony_tolerance)" : ""
    print(io, "NSGAIIConfig(pop=$(config.population_size), gen=$(config.max_generations), depth=$(config.min_depth):$(config.max_depth)$nodes_str$parsimony_str)")
end

# ───────────────────────────────────────────────────────────────────────────────
# Result Structure
# ───────────────────────────────────────────────────────────────────────────────

"""
    NSGAIIResult

Results from an NSGA-II optimization run.

# Fields
- `pareto_front::Vector{Individual}`: Non-dominated individuals from final population
- `population::Vector{Individual}`: Final population
- `generations::Int`: Number of generations run
- `history::Vector{Dict}`: Per-generation statistics
- `best_per_objective::Vector{Individual}`: Best individual for each objective
"""
struct NSGAIIResult
    pareto_front::Vector{Individual}
    population::Vector{Individual}
    generations::Int
    history::Vector{Dict{Symbol,Any}}
    best_per_objective::Vector{Individual}
end

function Base.show(io::IO, result::NSGAIIResult)
    print(io, "NSGAIIResult(generations=$(result.generations), pareto_size=$(length(result.pareto_front)))")
end

"""
    get_best(result::NSGAIIResult, objective_index::Int=1) -> Individual

Get the best individual for a specific objective.
"""
function get_best(result::NSGAIIResult, objective_index::Int=1)
    if objective_index <= length(result.best_per_objective)
        return result.best_per_objective[objective_index]
    else
        error("Objective index $objective_index out of range (have $(length(result.best_per_objective)) objectives)")
    end
end

"""
    get_pareto_front(result::NSGAIIResult) -> Vector{Individual}

Get all non-dominated individuals.
"""
get_pareto_front(result::NSGAIIResult) = result.pareto_front
