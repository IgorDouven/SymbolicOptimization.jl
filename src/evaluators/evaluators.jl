#=
Evaluators Module
=================

Pre-built evaluation functions for common symbolic policy problems.

These evaluators handle the mapping from (tree, environment) → objective values
for different problem types:

- `discrimination_evaluator`: For problems where formulas produce scores that
  should discriminate between positive/negative cases (AUC-based)
  
- `sequential_evaluator`: For problems where formulas are applied sequentially
  with state accumulation (e.g., belief updating)
  
- `calibration_evaluator`: For probability aggregation problems where we care
  about calibration and accuracy

Users can also define custom evaluators following the same pattern:
    evaluator(tree, env) -> Vector{Float64} of objective values
=#

module Evaluators

using Statistics: mean, std, var, cor
using Random: shuffle
import Random

export discrimination_evaluator, sequential_evaluator, calibration_evaluator
export compute_auc, compute_brier, compute_log_score

# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_auc(scores, labels)

Compute Area Under the ROC Curve for binary classification.
Higher scores should indicate positive class (label = true).
"""
function compute_auc(scores::Vector{<:Real}, labels::Vector{Bool})
    n = length(scores)
    n == 0 && return 0.5
    
    # Handle NaN/Inf scores
    valid_idx = findall(isfinite.(scores))
    if length(valid_idx) < 2
        return 0.5
    end
    
    scores = scores[valid_idx]
    labels = labels[valid_idx]
    
    # Count positives and negatives
    n_pos = sum(labels)
    n_neg = length(labels) - n_pos
    
    (n_pos == 0 || n_neg == 0) && return 0.5
    
    # Sort by score descending
    order = sortperm(scores, rev=true)
    sorted_labels = labels[order]
    
    # Compute AUC via Mann-Whitney U statistic
    # AUC = P(score_pos > score_neg)
    cum_neg = 0
    auc_sum = 0.0
    
    for label in sorted_labels
        if !label
            cum_neg += 1
        else
            auc_sum += cum_neg
        end
    end
    
    # AUC = 1 - (sum of ranks of positives) normalized
    return 1.0 - auc_sum / (n_pos * n_neg)
end

"""
    compute_brier(predictions, truth)

Compute Brier score: mean((pred - truth)^2)
"""
function compute_brier(predictions::Vector{<:Real}, truth::Vector{<:Real})
    valid_idx = findall(isfinite.(predictions))
    if isempty(valid_idx)
        return 1.0
    end
    return mean((predictions[valid_idx] .- truth[valid_idx]).^2)
end

"""
    compute_log_score(predictions, truth)

Compute logarithmic score: -mean(truth*log(pred) + (1-truth)*log(1-pred))
"""
function compute_log_score(predictions::Vector{<:Real}, truth::Vector{<:Real})
    valid_idx = findall(isfinite.(predictions))
    if isempty(valid_idx)
        return Inf
    end
    
    preds = clamp.(predictions[valid_idx], 1e-10, 1 - 1e-10)
    t = truth[valid_idx]
    
    return -mean(t .* log.(preds) .+ (1 .- t) .* log.(1 .- preds))
end

# ─────────────────────────────────────────────────────────────────────────────
# Discrimination Evaluator (AUC-based)
# ─────────────────────────────────────────────────────────────────────────────

"""
    discrimination_evaluator(; simulator, n_simulations=1000, objectives=[:auc, :complexity])

Create an evaluator for discrimination problems where formulas produce scores
that should separate positive from negative cases.

# Arguments
- `simulator`: Function `(rng) -> (inputs::Dict{Symbol,Float64}, label::Bool)`
  that generates one trial with input variables and ground truth label
- `n_simulations`: Number of simulation trials per evaluation
- `objectives`: Which objectives to compute. Options:
  - `:auc` - Area under ROC curve (maximized, so we return 1-AUC)
  - `:complexity` - Number of nodes in tree
  - `:correlation` - Correlation between scores and labels

# Returns
A function `(tree, env) -> Vector{Float64}` suitable for use in policy_problem.

# Example
```julia
# Simulator for confirmation tracking
function conf_simulator(rng)
    # Generate random probability setup
    ...
    inputs = Dict(:pH => pH, :pE => pE, :pH_E => pH_E, ...)
    label = H_is_true
    return (inputs, label)
end

evaluator = discrimination_evaluator(
    simulator = conf_simulator,
    n_simulations = 1000,
    objectives = [:auc, :complexity]
)
```
"""
function discrimination_evaluator(;
    simulator::Function,
    n_simulations::Int = 1000,
    objectives::Vector{Symbol} = [:auc, :complexity]
)
    return (tree, env, evaluate_fn, count_nodes_fn) -> begin
        rng = get(env, :rng, Random.default_rng())
        var_names = env[:var_names]
        
        scores = Float64[]
        labels = Bool[]
        
        for _ in 1:n_simulations
            inputs, label = simulator(rng)
            
            # Build context from inputs
            ctx = Dict{Symbol, Any}()
            for (name, val) in inputs
                ctx[name] = val
            end
            
            # Evaluate tree
            score = evaluate_fn(tree, ctx)
            
            if isfinite(score)
                push!(scores, score)
                push!(labels, label)
            end
        end
        
        # Compute requested objectives
        obj_values = Float64[]
        
        for obj in objectives
            if obj == :auc
                auc = length(scores) >= 10 ? compute_auc(scores, labels) : 0.5
                push!(obj_values, 1.0 - auc)  # Minimize 1-AUC
            elseif obj == :complexity
                push!(obj_values, Float64(count_nodes_fn(tree)))
            elseif obj == :correlation
                if length(scores) >= 10
                    corr = cor(scores, Float64.(labels))
                    push!(obj_values, isfinite(corr) ? 1.0 - abs(corr) : 1.0)
                else
                    push!(obj_values, 1.0)
                end
            else
                error("Unknown objective: $obj")
            end
        end
        
        return obj_values
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Sequential Evaluator (for belief updating, etc.)
# ─────────────────────────────────────────────────────────────────────────────

"""
    sequential_evaluator(; sequences, target_key, objectives=[:mse, :complexity])

Create an evaluator for sequential problems where formulas are applied step-by-step.

# Arguments
- `sequences`: Vector of sequences, each sequence is Vector{Dict{Symbol,Any}}
  representing steps with input variables. Must include `target_key` for comparison.
- `target_key`: Symbol for the target value to predict at each step
- `state_keys`: Symbols that carry state between steps (updated with formula output)
- `objectives`: Which objectives to compute

# Example (belief updating)
```julia
# Each sequence is a series of (prior, evidence, posterior) steps
sequences = [
    [Dict(:prior => 0.5, :likelihood => 0.8, :target => 0.67), ...],
    ...
]

evaluator = sequential_evaluator(
    sequences = sequences,
    target_key = :target,
    objectives = [:mse, :complexity]
)
```
"""
function sequential_evaluator(;
    sequences::Vector{<:Vector},
    target_key::Symbol,
    objectives::Vector{Symbol} = [:mse, :complexity]
)
    return (tree, env, evaluate_fn, count_nodes_fn) -> begin
        all_predictions = Float64[]
        all_targets = Float64[]
        
        for seq in sequences
            for step in seq
                # Evaluate formula on this step's inputs
                pred = evaluate_fn(tree, step)
                target = step[target_key]
                
                if isfinite(pred) && isfinite(target)
                    push!(all_predictions, pred)
                    push!(all_targets, target)
                end
            end
        end
        
        # Compute objectives
        obj_values = Float64[]
        
        for obj in objectives
            if obj == :mse
                mse = isempty(all_predictions) ? 1.0 : 
                      mean((all_predictions .- all_targets).^2)
                push!(obj_values, mse)
            elseif obj == :mae
                mae = isempty(all_predictions) ? 1.0 :
                      mean(abs.(all_predictions .- all_targets))
                push!(obj_values, mae)
            elseif obj == :complexity
                push!(obj_values, Float64(count_nodes_fn(tree)))
            elseif obj == :correlation
                if length(all_predictions) >= 10
                    corr = cor(all_predictions, all_targets)
                    push!(obj_values, isfinite(corr) ? 1.0 - corr : 1.0)
                else
                    push!(obj_values, 1.0)
                end
            else
                error("Unknown objective: $obj")
            end
        end
        
        return obj_values
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Calibration Evaluator (for probability aggregation)
# ─────────────────────────────────────────────────────────────────────────────

"""
    calibration_evaluator(; objectives=[:brier, :complexity])

Create an evaluator for probability aggregation problems.

Expects environment to contain:
- `:X` - Matrix of forecaster predictions (rows = items, cols = forecasters)
- `:y` - Vector of ground truth (0/1)
- `:var_names` - Variable names for each forecaster column

# Objectives
- `:brier` - Brier score
- `:log_score` - Logarithmic scoring rule
- `:accuracy` - Classification accuracy at 0.5 threshold
- `:complexity` - Number of nodes
"""
function calibration_evaluator(;
    objectives::Vector{Symbol} = [:brier, :complexity]
)
    return (tree, env, evaluate_fn, count_nodes_fn) -> begin
        X = env[:X]
        y = env[:y]
        var_names = env[:var_names]
        
        predictions = Float64[]
        
        for i in axes(X, 1)
            ctx = Dict(var_names[j] => X[i, j] for j in eachindex(var_names))
            pred = evaluate_fn(tree, ctx)
            push!(predictions, isfinite(pred) ? clamp(pred, 0.0, 1.0) : 0.5)
        end
        
        # Compute objectives
        obj_values = Float64[]
        
        for obj in objectives
            if obj == :brier
                push!(obj_values, compute_brier(predictions, y))
            elseif obj == :log_score
                push!(obj_values, compute_log_score(predictions, y))
            elseif obj == :accuracy
                acc = mean((predictions .>= 0.5) .== (y .>= 0.5))
                push!(obj_values, 1.0 - acc)  # Minimize 1-accuracy
            elseif obj == :complexity
                push!(obj_values, Float64(count_nodes_fn(tree)))
            else
                error("Unknown objective: $obj")
            end
        end
        
        return obj_values
    end
end

end # module Evaluators
