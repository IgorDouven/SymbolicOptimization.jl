#=
Belief Updating Heuristic Discovery
====================================

This example replicates results from:
    Douven, I. "Ecological Rationality from the Ground Up"

The task: Discover belief-updating heuristics for a coin bias identification problem.
An agent observes coin flips and must estimate which of 11 bias hypotheses (0.0, 0.1, ..., 1.0)
is closest to the true bias.

The evolutionary search optimizes three objectives:
1. Speed: How quickly the rule converges to high confidence in the correct hypothesis
2. Accuracy: Brier score (how close probability estimates are to truth)
3. Complexity: Number of nodes in the expression tree

Expected discovery: A family of simple "follow the leader" heuristics:
    normalize(add_ibe_bonus(probs, c))
which boost the hypothesis closest to the running data mean by a small constant c.

Run from package directory:
    julia --project=. examples/belief_updating_discovery.jl
=#

using SymbolicOptimization
using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf

println("="^70)
println("Belief Updating Heuristic Discovery")
println("Replicating: Douven (2025) 'Ecological Rationality from the Ground Up'")
println("="^70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Problem Constants
# ─────────────────────────────────────────────────────────────────────────────

const NUM_HYPOTHESES = 11
const BIASES = collect(range(0.0, 1.0, length=NUM_HYPOTHESES))
const PRIOR_UNIFORM = fill(1.0/NUM_HYPOTHESES, NUM_HYPOTHESES)
const LIKELIHOOD_HEADS = copy(BIASES)
const LIKELIHOOD_TAILS = 1.0 .- BIASES
const SURPRISE_HEADS = -log.(LIKELIHOOD_HEADS .+ 1e-10)
const SURPRISE_TAILS = -log.(LIKELIHOOD_TAILS .+ 1e-10)

const NUM_TOSSES = 250
const CONFIDENCE_THRESHOLD = 0.75
const HEAD_START = 30
const NUM_EVAL_SCENARIOS = 100  # Scenarios per fitness evaluation (higher = less variance, slower)

println("\n1. Problem Setup")
println("-"^50)
println("Hypotheses: $NUM_HYPOTHESES bias values from 0.0 to 1.0")
println("Trial length: $NUM_TOSSES coin flips")
println("Confidence threshold: $CONFIDENCE_THRESHOLD")
println("Head start period: $HEAD_START updates")
println("Evaluation scenarios: $NUM_EVAL_SCENARIOS per fitness call (fresh each evaluation)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom Operators for Belief Updating
# ─────────────────────────────────────────────────────────────────────────────

println("\n2. Defining Custom Operators")
println("-"^50)

# Helper: ensure vector has correct length
to_vec(x::AbstractVector) = length(x) == NUM_HYPOTHESES ? x : fill(mean(x), NUM_HYPOTHESES)
to_vec(x::Number) = fill(Float64(x), NUM_HYPOTHESES)

# Safe scalar operations
safe_add(a, b) = Float64(a) + Float64(b)
safe_sub(a, b) = Float64(a) - Float64(b)
safe_mul(a, b) = Float64(a) * Float64(b)
safe_div_s(a, b) = abs(b) < 1e-10 ? Float64(a) : Float64(a) / Float64(b)
safe_log_s(a) = log(abs(Float64(a)) + 1e-10)
safe_exp_s(a) = exp(clamp(Float64(a), -10.0, 10.0))

# Vector operations
v_add(a, b) = to_vec(a) .+ to_vec(b)
v_sub(a, b) = to_vec(a) .- to_vec(b)
v_mul(a, b) = to_vec(a) .* to_vec(b)
v_div(a, b) = to_vec(a) ./ (to_vec(b) .+ 1e-10)
v_pow(a, b) = (abs.(to_vec(a)) .+ 1e-10) .^ clamp(Float64(b), -5.0, 5.0)
sv_mul(s, v) = Float64(s) .* to_vec(v)

# Reductions
safe_mean_v(v) = mean(to_vec(v))
safe_dot_v(a, b) = dot(to_vec(a), to_vec(b))

# normalize: ensure probabilities sum to 1
normalize_vec(v) = begin
    vec = to_vec(v)
    vec = max.(vec, 0.0)
    s = sum(vec)
    s > 0 ? vec ./ s : PRIOR_UNIFORM
end

# weighted_sum: w*a + (1-w)*b
weighted_sum_op(w, a, b) = begin
    weight = clamp(Float64(w), 0.0, 1.0)
    weight .* to_vec(a) .+ (1.0 - weight) .* to_vec(b)
end

println("Defined: normalize, add_ibe_bonus, select_by_data, weighted_sum, v_mul, etc.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Grammar Definition
# ─────────────────────────────────────────────────────────────────────────────

println("\n3. Building Grammar")
println("-"^50)

# IMPORTANT: Following the paper's methodology, we enforce that `normalize` is 
# ALWAYS the root node. We do this by:
# 1. NOT including `normalize` in the grammar's operators
# 2. Having the grammar produce "raw" vectors (type :RawVector)
# 3. Wrapping all trees with `normalize` during evaluation and display
#
# This ensures every rule outputs valid probabilities.

# For context-aware operators, we need placeholder functions for the grammar.
# The actual context-aware implementations are provided via EvalContext operators dict.

# Placeholder for add_ibe_bonus: (Vector, Scalar) -> Vector
add_ibe_bonus_placeholder(v, bonus) = to_vec(v)  # Identity (placeholder)

# Placeholder for select_by_data: (Vector, Vector) -> Vector
select_by_data_placeholder(a, b) = to_vec(a)  # Just returns first arg (placeholder)

# Register all operator implementations in SAFE_IMPLEMENTATIONS
SAFE_IMPLEMENTATIONS[:weighted_sum] = weighted_sum_op
SAFE_IMPLEMENTATIONS[:v_add] = v_add
SAFE_IMPLEMENTATIONS[:v_sub] = v_sub
SAFE_IMPLEMENTATIONS[:v_mul] = v_mul
SAFE_IMPLEMENTATIONS[:v_pow] = v_pow
SAFE_IMPLEMENTATIONS[:sv_mul] = sv_mul
SAFE_IMPLEMENTATIONS[:safe_mean_v] = safe_mean_v
SAFE_IMPLEMENTATIONS[:safe_dot_v] = safe_dot_v
SAFE_IMPLEMENTATIONS[:safe_log_s] = safe_log_s
SAFE_IMPLEMENTATIONS[:safe_exp_s] = safe_exp_s
SAFE_IMPLEMENTATIONS[:safe_div_s] = safe_div_s
SAFE_IMPLEMENTATIONS[:add_ibe_bonus] = add_ibe_bonus_placeholder
SAFE_IMPLEMENTATIONS[:select_by_data] = select_by_data_placeholder

grammar = Grammar(
    types = [:Scalar, :RawVector],  # RawVector = unnormalized, will be wrapped with normalize
    
    variables = [
        :probs => :RawVector,        # Current belief distribution
        :likelihood_h => :RawVector, # P(heads|H_i)
        :likelihood_t => :RawVector, # P(tails|H_i)
        :prior => :RawVector,        # Uniform prior
        :data_mean => :Scalar,       # Running mean of observations
        :data_std => :Scalar,        # Running std of observations
        :n_tosses => :Scalar,        # Number of observations so far
        :max_prob => :Scalar,        # Maximum probability in current distribution
        :entropy => :Scalar,         # Entropy of current distribution
    ],
    
    operators = [
        # The key operator: add_ibe_bonus (context-dependent)
        (:add_ibe_bonus, [:RawVector, :Scalar] => :RawVector),
        
        # Conditional selection (context-dependent)
        (:select_by_data, [:RawVector, :RawVector] => :RawVector),
        
        # NOTE: normalize is NOT in the grammar - it's implicitly at the root
        
        # Weighted combination
        (:weighted_sum, [:Scalar, :RawVector, :RawVector] => :RawVector),
        
        # Vector operations
        (:v_add, [:RawVector, :RawVector] => :RawVector),
        (:v_sub, [:RawVector, :RawVector] => :RawVector),
        (:v_mul, [:RawVector, :RawVector] => :RawVector),
        (:v_pow, [:RawVector, :Scalar] => :RawVector),
        (:sv_mul, [:Scalar, :RawVector] => :RawVector),
        
        # Reductions
        (:safe_mean_v, [:RawVector] => :Scalar),
        (:safe_dot_v, [:RawVector, :RawVector] => :Scalar),
        
        # Scalar operations
        (:+, [:Scalar, :Scalar] => :Scalar),
        (:-, [:Scalar, :Scalar] => :Scalar),
        (:*, [:Scalar, :Scalar] => :Scalar),
        (:safe_div_s, [:Scalar, :Scalar] => :Scalar),
        (:safe_log_s, [:Scalar] => :Scalar),
        (:safe_exp_s, [:Scalar] => :Scalar),
    ],
    
    constant_types = [:Scalar],
    constant_range = (0.0, 0.5),  # Small positive constants for bonuses
    output_type = :RawVector,     # Grammar produces raw vectors; normalize is added implicitly
)

println("Grammar created with $(num_operators(grammar)) operators")
println("Output type: RawVector (will be wrapped with normalize)")
println("NOTE: normalize is implicit at root - all rules are: normalize(<tree>)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Context-Aware Evaluation System
# ─────────────────────────────────────────────────────────────────────────────

println("\n4. Setting Up Context-Aware Evaluation")
println("-"^50)

"""
State for tracking belief updating across a trial.
"""
mutable struct BeliefState
    probs::Vector{Float64}
    data::Vector{Bool}
    n_tosses::Int
    data_mean::Float64
    data_std::Float64
    max_prob::Float64
    entropy::Float64
    is_heads::Bool
end

function update_belief_state!(state::BeliefState)
    if state.n_tosses > 0
        observations = state.data[1:state.n_tosses]
        state.data_mean = mean(observations)
        state.data_std = state.n_tosses > 1 ? std(observations) : 0.0
        state.is_heads = state.data[state.n_tosses]
    else
        state.data_mean = 0.5
        state.data_std = 0.0
        state.is_heads = true
    end
    state.max_prob = maximum(state.probs)
    state.entropy = -sum(p * log(p + 1e-10) for p in state.probs)
end

"""
Create an EvalContext for belief updating from a BeliefState.
Custom operators (add_ibe_bonus, select_by_data) receive the full context.
"""
function make_belief_context(state::BeliefState)
    bindings = Dict{Symbol, Any}(
        :probs => state.probs,
        :likelihood_h => LIKELIHOOD_HEADS,
        :likelihood_t => LIKELIHOOD_TAILS,
        :prior => PRIOR_UNIFORM,
        :data_mean => state.data_mean,
        :data_std => state.data_std,
        :n_tosses => Float64(state.n_tosses),
        :max_prob => state.max_prob,
        :entropy => state.entropy,
    )
    
    # Context-aware operators (normalize is NOT here - it's implicit at root)
    operators = Dict{Symbol, Function}(
        :add_ibe_bonus => (v, bonus, ctx) -> begin
            vec = to_vec(v)
            n = ctx[:n_tosses]
            if n > 0
                bonus_amount = clamp(Float64(bonus), 0.0, 1.0)
                dm = ctx[:data_mean]
                _, best_idx = findmin(abs.(BIASES .- dm))
                result = copy(vec)
                result[best_idx] += bonus_amount
                return result
            end
            return vec
        end,
        
        :select_by_data => (a, b, ctx) -> begin
            state.is_heads ? to_vec(a) : to_vec(b)
        end,
        
        :weighted_sum => (w, a, b, ctx) -> weighted_sum_op(w, a, b),
        :v_add => (a, b, ctx) -> v_add(a, b),
        :v_sub => (a, b, ctx) -> v_sub(a, b),
        :v_mul => (a, b, ctx) -> v_mul(a, b),
        :v_pow => (a, b, ctx) -> v_pow(a, b),
        :sv_mul => (s, v, ctx) -> sv_mul(s, v),
        :safe_mean_v => (v, ctx) -> safe_mean_v(v),
        :safe_dot_v => (a, b, ctx) -> safe_dot_v(a, b),
        :+ => (a, b, ctx) -> safe_add(a, b),
        :- => (a, b, ctx) -> safe_sub(a, b),
        :* => (a, b, ctx) -> safe_mul(a, b),
        :safe_div_s => (a, b, ctx) -> safe_div_s(a, b),
        :safe_log_s => (a, ctx) -> safe_log_s(a),
        :safe_exp_s => (a, ctx) -> safe_exp_s(a),
    )
    
    return EvalContext(bindings, operators)
end

"""
Apply a belief updating rule for one step.
NOTE: The tree is implicitly wrapped with normalize - this ensures
all rules output valid probability distributions.
"""
function apply_rule(tree::AbstractNode, probs::Vector{Float64}, data::Vector{Bool}, t::Int)
    state = BeliefState(copy(probs), data, t, 0.5, 0.0, 0.0, 0.0, true)
    update_belief_state!(state)
    ctx = make_belief_context(state)
    
    result = try
        evaluate(tree, ctx)
    catch e
        return probs  # Return unchanged on error
    end
    
    # Implicit normalize wrapper - always normalize the result
    isa(result, Number) && (result = probs .* Float64(result))
    result = to_vec(result)
    result = normalize_vec(result)  # This is the implicit normalize at root
    
    return result
end

"""
Helper to display a tree with implicit normalize wrapper.
"""
function display_with_normalize(tree::AbstractNode)
    return "normalize($(node_to_string(tree)))"
end

println("Context-aware evaluation system ready")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Objective Functions
# ─────────────────────────────────────────────────────────────────────────────

println("\n5. Defining Objective Functions")
println("-"^50)

function brier_score(probs::Vector{Float64}, true_idx::Int)
    sum(probs) ≈ 0.0 && return 1.0
    (1.0 - probs[true_idx])^2 + sum(probs[i]^2 for i in eachindex(probs) if i != true_idx)
end

"""
Speed objective: How quickly does the rule converge to correct answer?
Lower is better. Penalties for wrong convergence or timeout.

NOTE: Uses fresh random scenarios each call to avoid overfitting.
"""
function speed_objective_func(tree::AbstractNode, data::Dict)
    n_scenarios = get(data, :n_scenarios, NUM_EVAL_SCENARIOS)
    # Fresh RNG each evaluation to avoid overfitting to fixed scenarios
    rng = MersenneTwister()
    
    total_speed = 0.0
    for _ in 1:n_scenarios
        true_bias = rand(rng)
        coin_data = rand(rng, Bernoulli(true_bias), NUM_TOSSES)
        probs = copy(PRIOR_UNIFORM)
        true_idx = round(Int, true_bias * (NUM_HYPOTHESES - 1)) + 1
        
        # Head start period
        for t in 1:HEAD_START
            probs = apply_rule(tree, probs, coin_data, t)
            (any(!isfinite, probs) || sum(probs) ≈ 0) && (probs = copy(PRIOR_UNIFORM))
        end
        
        # Main evaluation
        converged = false
        for t in (HEAD_START + 1):NUM_TOSSES
            probs = apply_rule(tree, probs, coin_data, t)
            (any(!isfinite, probs) || sum(probs) ≈ 0) && (probs = copy(PRIOR_UNIFORM))
            
            max_p, max_idx = findmax(probs)
            if max_p > CONFIDENCE_THRESHOLD
                if max_idx == true_idx
                    total_speed += Float64(t)  # Good convergence
                else
                    total_speed += Float64(NUM_TOSSES * 3)  # Wrong answer penalty
                end
                converged = true
                break
            end
        end
        
        if !converged
            total_speed += Float64(NUM_TOSSES * 2)  # Timeout penalty
        end
    end
    
    return total_speed / n_scenarios
end

"""
Brier objective: Weighted Brier score over the trial.
Lower is better.

NOTE: Uses fresh random scenarios each call to avoid overfitting.
"""
function brier_objective_func(tree::AbstractNode, data::Dict)
    n_scenarios = get(data, :n_scenarios, NUM_EVAL_SCENARIOS)
    # Fresh RNG each evaluation to avoid overfitting to fixed scenarios
    rng = MersenneTwister()
    
    total_brier = 0.0
    for _ in 1:n_scenarios
        true_bias = rand(rng)
        coin_data = rand(rng, Bernoulli(true_bias), NUM_TOSSES)
        probs = copy(PRIOR_UNIFORM)
        true_idx = round(Int, true_bias * (NUM_HYPOTHESES - 1)) + 1
        
        trial_brier = 0.0
        total_weight = 0.0
        for t in 1:NUM_TOSSES
            probs = apply_rule(tree, probs, coin_data, t)
            (any(!isfinite, probs) || sum(probs) ≈ 0) && (probs = copy(PRIOR_UNIFORM))
            
            w = log(t + 1)  # Log-weighted
            trial_brier += brier_score(probs, true_idx) * w
            total_weight += w
        end
        
        total_brier += trial_brier / total_weight
    end
    
    return total_brier / n_scenarios
end

# Create objective objects
speed_obj = custom_objective(:speed, speed_objective_func; minimize=true)
brier_obj = custom_objective(:brier, brier_objective_func; minimize=true)
complexity_obj = complexity_objective()

objectives = [speed_obj, brier_obj, complexity_obj]
println("Objectives: Speed (convergence time), Brier score, Complexity")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Baseline Evaluation
# ─────────────────────────────────────────────────────────────────────────────

println("\n6. Baseline Rule Comparison")
println("-"^50)

function bayes_update(probs::Vector{Float64}, data::Vector{Bool}, t::Int)
    likelihood = data[t] ? LIKELIHOOD_HEADS : LIKELIHOOD_TAILS
    posterior = probs .* likelihood
    s = sum(posterior)
    s > 0 ? posterior ./ s : probs
end

function follow_leader_update(probs::Vector{Float64}, data::Vector{Bool}, t::Int; bonus::Float64=0.05)
    _, best_idx = findmin(abs.(BIASES .- mean(data[1:t])))
    result = copy(probs)
    result[best_idx] += bonus
    result ./ sum(result)
end

rng = MersenneTwister(42)

function eval_baseline(update_fn, name, rng; n=100)
    speeds, briers = Float64[], Float64[]
    for _ in 1:n
        bias = rand(rng)
        data = rand(rng, Bernoulli(bias), NUM_TOSSES)
        true_idx = round(Int, bias * (NUM_HYPOTHESES - 1)) + 1
        
        probs = copy(PRIOR_UNIFORM)
        conv_time, conf_idx = NUM_TOSSES + 1, -1
        for t in 1:HEAD_START; probs = update_fn(probs, data, t); end
        for t in (HEAD_START + 1):NUM_TOSSES
            probs = update_fn(probs, data, t)
            max_p, max_idx = findmax(probs)
            if max_p > CONFIDENCE_THRESHOLD
                conv_time, conf_idx = t, max_idx
                break
            end
        end
        push!(speeds, conf_idx == true_idx ? conv_time : conf_idx != -1 ? NUM_TOSSES * 3 : NUM_TOSSES * 2)
        
        probs = copy(PRIOR_UNIFORM)
        total_w, total = 0.0, 0.0
        for t in 1:NUM_TOSSES
            probs = update_fn(probs, data, t)
            w = log(t + 1)
            total += brier_score(probs, true_idx) * w
            total_w += w
        end
        push!(briers, total / total_w)
    end
    @printf("  %-25s Speed: %6.1f  Brier: %.4f\n", name, mean(speeds), mean(briers))
end

eval_baseline(bayes_update, "Bayes' rule", rng)
eval_baseline((p,d,t) -> follow_leader_update(p,d,t,bonus=0.04), "Follow leader (c=0.04)", rng)
eval_baseline((p,d,t) -> follow_leader_update(p,d,t,bonus=0.09), "Follow leader (c=0.09)", rng)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Build Seeds (Optional - NOT including target solutions)
# ─────────────────────────────────────────────────────────────────────────────

println("\n7. Building Seed Population (optional)")
println("-"^50)

"""
Build initial population with basic building blocks.
IMPORTANT: We do NOT include the target follow-leader heuristics here.
The goal is to DISCOVER them through search, replicating the paper's methodology.

NOTE: normalize is implicit at root, so we don't include it in trees.
All trees are effectively: normalize(<tree>)
"""
function build_belief_seeds()
    seeds = Individual[]
    probs = Variable(:probs, :RawVector)
    likelihood_h = Variable(:likelihood_h, :RawVector)
    likelihood_t = Variable(:likelihood_t, :RawVector)
    prior = Variable(:prior, :RawVector)
    
    # 1. Bayes-like: normalize(probs * likelihood_h)
    # Tree is just: v_mul(probs, likelihood_h) - normalize is implicit
    bayes_like = FunctionNode(:v_mul, probs, likelihood_h)
    push!(seeds, Individual(bayes_like))
    
    # 2. Full Bayesian update with select_by_data
    # normalize(select_by_data(probs*lh, probs*lt))
    select_bayes = FunctionNode(:select_by_data,
        FunctionNode(:v_mul, probs, likelihood_h),
        FunctionNode(:v_mul, probs, likelihood_t))
    push!(seeds, Individual(select_bayes))
    
    # 3. Just return probs (no update - baseline)
    # normalize(probs)
    no_update = probs
    push!(seeds, Individual(no_update))
    
    # 4. Prior-weighted combination (not using add_ibe_bonus)
    # normalize(weighted_sum(0.95, probs, prior))
    weighted_prior = FunctionNode(:weighted_sum,
        Constant(0.95),
        probs,
        prior)
    push!(seeds, Individual(weighted_prior))
    
    # NOTE: We intentionally do NOT seed any add_ibe_bonus solutions.
    # The goal is to discover the follow-leader heuristic through search.
    
    return seeds
end

seeds = build_belief_seeds()
println("Created $(length(seeds)) seed individuals (NO follow-leader seeds)")
println("Target heuristic must be DISCOVERED through evolution")
println("NOTE: All trees implicitly wrapped with normalize")

# Evaluate seeds
println("\nSeed evaluation:")
eval_data = Dict{Symbol, Any}(
    :n_scenarios => NUM_EVAL_SCENARIOS,
)
for (i, seed) in enumerate(seeds)
    speed = speed_objective_func(seed.tree, eval_data)
    brier = brier_objective_func(seed.tree, eval_data)
    nodes = count_nodes(seed.tree) + 1  # +1 for implicit normalize
    expr = display_with_normalize(seed.tree)
    if length(expr) > 50
        expr = expr[1:47] * "..."
    end
    @printf("  Seed %2d: Speed=%6.1f  Brier=%.4f  nodes=%2d  %s\n", 
            i, speed, brier, nodes, expr)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Run NSGA-II Optimization
# ─────────────────────────────────────────────────────────────────────────────

println("\n8. Running NSGA-II Optimization")
println("-"^50)

# Configuration toggles
USE_SEEDS = false  # Set to false for pure discovery (recommended for replication)
QUICK_RUN = true   # Set to false for full optimization

if QUICK_RUN
    config = NSGAIIConfig(
        population_size = 200,
        max_generations = 50,
        max_depth = 4,
        max_nodes = 12,
        tournament_size = 3,
        crossover_prob = 0.7,
        mutation_prob = 0.8,
        elite_fraction = 0.05,
        parsimony_tolerance = 0.05,
        simplify_prob = 0.1,
        early_stop_generations = 25,
        verbose = true,
    )
    println("Mode: QUICK RUN (for testing)")
else
    config = NSGAIIConfig(
        population_size = 500,
        max_generations = 150,
        max_depth = 5,
        max_nodes = 15,
        tournament_size = 3,
        crossover_prob = 0.7,
        mutation_prob = 0.85,
        elite_fraction = 0.02,
        parsimony_tolerance = 0.05,
        simplify_prob = 0.15,
        early_stop_generations = 50,
        verbose = true,
    )
    println("Mode: FULL OPTIMIZATION")
end

if USE_SEEDS
    initial_pop = seeds
    println("Using $(length(seeds)) seeds (Bayes-like baselines only)")
else
    initial_pop = nothing
    println("Pure random initialization (recommended for discovery)")
end

println("\nConfiguration:")
println("  Population: $(config.population_size)")
println("  Generations: $(config.max_generations)")
println("  Max depth: $(config.max_depth)")
println("  Max nodes: $(config.max_nodes)")

# Prepare data for objectives
opt_rng = MersenneTwister(12345)  # Only used for NSGA-II internals (selection, mutation)
data = Dict{Symbol, Any}(
    :n_scenarios => NUM_EVAL_SCENARIOS,
    :grammar => grammar,
)

println("\nStarting optimization...")
result = optimize(grammar, objectives, data;
    config = config,
    rng = opt_rng,
    initial_population = initial_pop
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Analyze Results
# ─────────────────────────────────────────────────────────────────────────────

println("\n9. Results Analysis")
println("-"^50)

# Get Pareto front
front = get_pareto_front(result)
println("Pareto front size (with duplicates): $(length(front))")

# Deduplicate by tree structure (same expression = same solution)
function deduplicate_front(front)
    seen = Set{String}()
    unique_inds = Individual[]
    for ind in front
        expr = node_to_string(ind.tree)
        if expr ∉ seen
            push!(seen, expr)
            push!(unique_inds, ind)
        end
    end
    return unique_inds
end

unique_front = deduplicate_front(collect(front))
println("Unique solutions: $(length(unique_front))")
println("NOTE: All expressions shown with implicit normalize wrapper")

# Collect and evaluate results
println("\nTop solutions by Speed (lower is better):")
sorted_by_speed = sort(unique_front, by=ind -> ind.objectives[1])
for (i, ind) in enumerate(sorted_by_speed[1:min(10, length(sorted_by_speed))])
    expr = display_with_normalize(ind.tree)
    if length(expr) > 55
        expr = expr[1:52] * "..."
    end
    @printf("  %2d. Speed=%6.1f  Brier=%.4f  nodes=%2d  %s\n",
            i, ind.objectives[1], ind.objectives[2], Int(ind.objectives[3])+1, expr)
end

println("\nTop solutions by Brier (lower is better):")
sorted_by_brier = sort(unique_front, by=ind -> ind.objectives[2])
for (i, ind) in enumerate(sorted_by_brier[1:min(10, length(sorted_by_brier))])
    expr = display_with_normalize(ind.tree)
    if length(expr) > 55
        expr = expr[1:52] * "..."
    end
    @printf("  %2d. Speed=%6.1f  Brier=%.4f  nodes=%2d  %s\n",
            i, ind.objectives[1], ind.objectives[2], Int(ind.objectives[3])+1, expr)
end

println("\nSimplest solutions (by node count):")
sorted_by_complexity = sort(unique_front, by=ind -> ind.objectives[3])
for (i, ind) in enumerate(sorted_by_complexity[1:min(10, length(sorted_by_complexity))])
    expr = display_with_normalize(ind.tree)
    if length(expr) > 55
        expr = expr[1:52] * "..."
    end
    @printf("  %2d. Speed=%6.1f  Brier=%.4f  nodes=%2d  %s\n",
            i, ind.objectives[1], ind.objectives[2], Int(ind.objectives[3])+1, expr)
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. Test Set Evaluation
# ─────────────────────────────────────────────────────────────────────────────

println("\n10. Test Set Evaluation")
println("-"^50)

# Use more scenarios for final test evaluation
test_data = Dict{Symbol, Any}(
    :n_scenarios => 200,  # More scenarios for reliable final evaluation
    :grammar => grammar,
)

println("Evaluating top solutions on test set (200 scenarios):")
println()

# Best by combined score (simple ranking)
function combined_rank(ind)
    # Normalize objectives roughly and sum
    speed_norm = ind.objectives[1] / 250.0  # Normalize by max tosses
    brier_norm = ind.objectives[2]          # Already 0-1 scale
    complexity_norm = ind.objectives[3] / 15.0  # Normalize by max nodes
    return speed_norm + brier_norm + 0.1 * complexity_norm
end

sorted_combined = sort(unique_front, by=combined_rank)

println("Best combined solutions (test set):")
for (i, ind) in enumerate(sorted_combined[1:min(5, length(sorted_combined))])
    test_speed = speed_objective_func(ind.tree, test_data)
    test_brier = brier_objective_func(ind.tree, test_data)
    expr = display_with_normalize(ind.tree)
    println()
    @printf("  %d. %s\n", i, expr)
    @printf("     Train: Speed=%6.1f  Brier=%.4f  Nodes=%d\n", 
            ind.objectives[1], ind.objectives[2], Int(ind.objectives[3])+1)
    @printf("     Test:  Speed=%6.1f  Brier=%.4f\n", test_speed, test_brier)
end

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "="^70)
println("Summary")
println("="^70)

println("""

EXPECTED DISCOVERY:
-------------------
The paper finds that evolutionary search consistently discovers simple
"follow the leader" heuristics of the form:

    normalize(add_ibe_bonus(probs, c))    where c ∈ [0.01, 0.12]

This rule boosts the hypothesis closest to the running data mean,
achieving FASTER convergence than Bayes' rule with COMPARABLE accuracy.

KEY DESIGN CHOICES (matching the paper):
----------------------------------------
• normalize is ALWAYS at the root (implicit, not in grammar)
• Target solution is NOT seeded - must be DISCOVERED
• Grammar generates "raw" vectors; normalize is applied automatically
• This ensures all rules output valid probability distributions

WHAT TO LOOK FOR:
-----------------
✓ Solutions containing add_ibe_bonus(probs, <small constant>) should emerge
✓ Simple solutions (3-4 nodes + normalize) should be competitive  
✓ Bayes-like solutions (v_mul with likelihood) may appear but be slower
✓ Best solutions should have Speed < 100 and Brier < 0.15

CONFIGURATION:
--------------
• QUICK_RUN = true  → Fast test (200 pop, 50 gen)
• QUICK_RUN = false → Full optimization (500 pop, 150 gen)
• USE_SEEDS = false → Pure discovery (recommended)
• USE_SEEDS = true  → Seed with Bayes-like baselines only
• NUM_EVAL_SCENARIOS = 100 → Scenarios per fitness evaluation (reduce for speed)
""")

println("="^70)
println("Example complete!")
println("="^70)
