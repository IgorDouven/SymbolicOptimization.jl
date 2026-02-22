#=
Constrained Confirmation Measure Search
=======================================

Search for confirmation measures that satisfy theoretical constraints:
1. Neutrality: measure = 0 when P(H|E) = P(H)
2. Directionality: positive when confirming, negative when disconfirming
3. Symmetric logicality: |max| = |min| at entailment extremes

Plus optimize for truth-tracking AUC.

The search grammar includes:
- Binary operators: +, -, *, /
- Unary operators: step (for piecewise measures), log (for Good-style measures)
- All relevant probability terms as variables

Usage:
    julia confirmation_constrained_search.jl
    
Make sure SymbolicOptimization is available (either installed or in the path).
=#

using Pkg

# Try to activate SymbolicOptimization - adjust path as needed
if isdir("SymbolicOptimization")
    Pkg.activate("SymbolicOptimization")
elseif isdir(joinpath(homedir(), "SymbolicOptimization"))
    Pkg.activate(joinpath(homedir(), "SymbolicOptimization"))
end

using SymbolicOptimization
using Random
using Statistics
using Distributions
using Printf

# Define step function for piecewise constructions
# step(x) = 1 if x ≥ 0, else 0
# This enables: step(cond) * A + (1 - step(cond)) * B  =  "if cond ≥ 0 then A else B"
step_func(x::Real) = x >= 0.0 ? 1.0 : 0.0

# Note: safe_log is already built into SymbolicOptimization - use `log` in the grammar

# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

"""
Generate a single confirmation trial from possible worlds framework.
Returns named tuple with all probability terms and the truth label.
"""
function generate_trial(rng::AbstractRNG, n_worlds::Int)
    @assert n_worlds > 3
    
    # Random sizes for H and E (non-tautological)
    n_H = rand(rng, 1:n_worlds-1)
    n_E = rand(rng, 1:n_worlds-1)
    
    # Random world assignments
    H_worlds = Set(randperm(rng, n_worlds)[1:n_H])
    E_worlds = Set(randperm(rng, n_worlds)[1:n_E])
    
    # Actual world must be in E (evidence is veridical)
    actual = rand(rng, collect(E_worlds))
    
    # Random probability mass
    π = rand(rng, Dirichlet(n_worlds, 1.0))
    
    # Compute all probability terms
    all_worlds = Set(1:n_worlds)
    notH_worlds = setdiff(all_worlds, H_worlds)
    notE_worlds = setdiff(all_worlds, E_worlds)
    
    pH = sum(π[w] for w in H_worlds)
    pE = sum(π[w] for w in E_worlds)
    pnotH = sum(π[w] for w in notH_worlds)
    pnotE = sum(π[w] for w in notE_worlds)
    
    pH_and_E = sum(π[w] for w in H_worlds ∩ E_worlds; init=0.0)
    pH_and_notE = sum(π[w] for w in H_worlds ∩ notE_worlds; init=0.0)
    pnotH_and_E = sum(π[w] for w in notH_worlds ∩ E_worlds; init=0.0)
    pnotH_and_notE = sum(π[w] for w in notH_worlds ∩ notE_worlds; init=0.0)
    
    # Conditional probabilities (with safety for division)
    pH_E = pE > 1e-10 ? pH_and_E / pE : 0.5
    pnotH_E = pE > 1e-10 ? pnotH_and_E / pE : 0.5
    pH_notE = pnotE > 1e-10 ? pH_and_notE / pnotE : 0.5
    pnotH_notE = pnotE > 1e-10 ? pnotH_and_notE / pnotE : 0.5
    pE_H = pH > 1e-10 ? pH_and_E / pH : 0.5
    pnotE_H = pH > 1e-10 ? pH_and_notE / pH : 0.5
    pE_notH = pnotH > 1e-10 ? pnotH_and_E / pnotH : 0.5
    pnotE_notH = pnotH > 1e-10 ? pnotH_and_notE / pnotH : 0.5
    
    # Truth label
    H_true = actual in H_worlds
    
    return (
        pH=pH, pE=pE, pnotH=pnotH, pnotE=pnotE,
        pH_and_E=pH_and_E, pH_and_notE=pH_and_notE,
        pnotH_and_E=pnotH_and_E, pnotH_and_notE=pnotH_and_notE,
        pH_E=pH_E, pnotH_E=pnotH_E, pH_notE=pH_notE, pnotH_notE=pnotH_notE,
        pE_H=pE_H, pnotE_H=pnotE_H, pE_notH=pE_notH, pnotE_notH=pnotE_notH,
        H_true=H_true
    )
end

"""
Generate trials specifically for neutrality testing (P(H|E) = P(H)).
"""
function generate_neutral_trial(rng::AbstractRNG)
    # Generate probabilities where H and E are independent
    pH = rand(rng) * 0.8 + 0.1  # P(H) ∈ [0.1, 0.9]
    pE = rand(rng) * 0.8 + 0.1  # P(E) ∈ [0.1, 0.9]
    
    # Independence: P(H∧E) = P(H)·P(E)
    pH_and_E = pH * pE
    pH_and_notE = pH * (1 - pE)
    pnotH_and_E = (1 - pH) * pE
    pnotH_and_notE = (1 - pH) * (1 - pE)
    
    pnotH = 1 - pH
    pnotE = 1 - pE
    
    # Conditionals (under independence)
    pH_E = pH          # P(H|E) = P(H) when independent
    pnotH_E = pnotH
    pH_notE = pH
    pnotH_notE = pnotH
    pE_H = pE
    pnotE_H = pnotE
    pE_notH = pE
    pnotE_notH = pnotE
    
    return (
        pH=pH, pE=pE, pnotH=pnotH, pnotE=pnotE,
        pH_and_E=pH_and_E, pH_and_notE=pH_and_notE,
        pnotH_and_E=pnotH_and_E, pnotH_and_notE=pnotH_and_notE,
        pH_E=pH_E, pnotH_E=pnotH_E, pH_notE=pH_notE, pnotH_notE=pnotH_notE,
        pE_H=pE_H, pnotE_H=pnotE_H, pE_notH=pE_notH, pnotE_notH=pnotE_notH
    )
end

"""
Generate trial where E entails H (P(H|E) = 1).
"""
function generate_entails_H_trial(rng::AbstractRNG)
    pH = rand(rng) * 0.8 + 0.1
    pE = rand(rng) * pH * 0.9 + 0.05  # P(E) < P(H) for E ⊨ H to be possible
    
    # E ⊨ H means all E-worlds are H-worlds
    pH_and_E = pE
    pnotH_and_E = 0.0
    pH_and_notE = pH - pE
    pnotH_and_notE = 1 - pH
    
    pnotH = 1 - pH
    pnotE = 1 - pE
    
    pH_E = 1.0
    pnotH_E = 0.0
    pH_notE = pnotE > 1e-10 ? pH_and_notE / pnotE : 0.5
    pnotH_notE = pnotE > 1e-10 ? pnotH_and_notE / pnotE : 0.5
    pE_H = pH > 1e-10 ? pE / pH : 0.5
    pnotE_H = pH > 1e-10 ? (pH - pE) / pH : 0.5
    pE_notH = 0.0
    pnotE_notH = 1.0
    
    return (
        pH=pH, pE=pE, pnotH=pnotH, pnotE=pnotE,
        pH_and_E=pH_and_E, pH_and_notE=pH_and_notE,
        pnotH_and_E=pnotH_and_E, pnotH_and_notE=pnotH_and_notE,
        pH_E=pH_E, pnotH_E=pnotH_E, pH_notE=pH_notE, pnotH_notE=pnotH_notE,
        pE_H=pE_H, pnotE_H=pnotE_H, pE_notH=pE_notH, pnotE_notH=pnotE_notH
    )
end

"""
Generate trial where E entails ¬H (P(H|E) = 0).
"""
function generate_entails_notH_trial(rng::AbstractRNG)
    pH = rand(rng) * 0.8 + 0.1
    pE = rand(rng) * (1 - pH) * 0.9 + 0.05  # P(E) < P(¬H) for E ⊨ ¬H to be possible
    
    # E ⊨ ¬H means all E-worlds are ¬H-worlds
    pH_and_E = 0.0
    pnotH_and_E = pE
    pH_and_notE = pH
    pnotH_and_notE = 1 - pH - pE
    
    pnotH = 1 - pH
    pnotE = 1 - pE
    
    pH_E = 0.0
    pnotH_E = 1.0
    pH_notE = pnotE > 1e-10 ? pH / pnotE : 0.5
    pnotH_notE = pnotE > 1e-10 ? (pnotH - pE) / pnotE : 0.5
    pE_H = 0.0
    pnotE_H = 1.0
    pE_notH = pnotH > 1e-10 ? pE / pnotH : 0.5
    pnotE_notH = pnotH > 1e-10 ? (pnotH - pE) / pnotH : 0.5
    
    return (
        pH=pH, pE=pE, pnotH=pnotH, pnotE=pnotE,
        pH_and_E=pH_and_E, pH_and_notE=pH_and_notE,
        pnotH_and_E=pnotH_and_E, pnotH_and_notE=pnotH_and_notE,
        pH_E=pH_E, pnotH_E=pnotH_E, pH_notE=pH_notE, pnotH_notE=pnotH_notE,
        pE_H=pE_H, pnotE_H=pnotE_H, pE_notH=pE_notH, pnotE_notH=pnotE_notH
    )
end

"""
Generate confirming trial (P(H|E) > P(H)).
"""
function generate_confirming_trial(rng::AbstractRNG, n_worlds::Int)
    # Keep generating until we get a confirming case
    for _ in 1:100
        trial = generate_trial(rng, n_worlds)
        if trial.pH_E > trial.pH + 0.01  # Clear confirmation
            return trial
        end
    end
    # Fallback: construct one directly
    return generate_entails_H_trial(rng)
end

"""
Generate disconfirming trial (P(H|E) < P(H)).
"""
function generate_disconfirming_trial(rng::AbstractRNG, n_worlds::Int)
    # Keep generating until we get a disconfirming case
    for _ in 1:100
        trial = generate_trial(rng, n_worlds)
        if trial.pH_E < trial.pH - 0.01  # Clear disconfirmation
            return trial
        end
    end
    # Fallback: construct one directly
    return generate_entails_notH_trial(rng)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Construction
# ═══════════════════════════════════════════════════════════════════════════════

function trial_to_dict(trial)
    Dict{Symbol,Float64}(
        :pH => trial.pH, :pE => trial.pE, :pnotH => trial.pnotH, :pnotE => trial.pnotE,
        :pH_and_E => trial.pH_and_E, :pH_and_notE => trial.pH_and_notE,
        :pnotH_and_E => trial.pnotH_and_E, :pnotH_and_notE => trial.pnotH_and_notE,
        :pH_E => trial.pH_E, :pnotH_E => trial.pnotH_E,
        :pH_notE => trial.pH_notE, :pnotH_notE => trial.pnotH_notE,
        :pE_H => trial.pE_H, :pnotE_H => trial.pnotE_H,
        :pE_notH => trial.pE_notH, :pnotE_notH => trial.pnotE_notH
    )
end

function create_dataset(rng::AbstractRNG; 
                        n_main::Int=3000,
                        n_neutral::Int=200,
                        n_entails_H::Int=200,
                        n_entails_notH::Int=200,
                        n_confirming::Int=200,
                        n_disconfirming::Int=200,
                        world_range::UnitRange{Int}=5:100)
    
    # Main training data (mixed world sizes)
    main_trials = [generate_trial(rng, rand(rng, world_range)) for _ in 1:n_main]
    main_X = [trial_to_dict(t) for t in main_trials]
    main_y = Float64[t.H_true ? 1.0 : 0.0 for t in main_trials]
    
    # Neutral trials (for neutrality constraint)
    neutral_trials = [generate_neutral_trial(rng) for _ in 1:n_neutral]
    neutral_X = [trial_to_dict(t) for t in neutral_trials]
    
    # Entailment trials (for logicality constraint)
    entails_H_trials = [generate_entails_H_trial(rng) for _ in 1:n_entails_H]
    entails_H_X = [trial_to_dict(t) for t in entails_H_trials]
    
    entails_notH_trials = [generate_entails_notH_trial(rng) for _ in 1:n_entails_notH]
    entails_notH_X = [trial_to_dict(t) for t in entails_notH_trials]
    
    # Confirming/disconfirming trials (for directionality constraint)
    confirming_trials = [generate_confirming_trial(rng, rand(rng, world_range)) for _ in 1:n_confirming]
    confirming_X = [trial_to_dict(t) for t in confirming_trials]
    
    disconfirming_trials = [generate_disconfirming_trial(rng, rand(rng, world_range)) for _ in 1:n_disconfirming]
    disconfirming_X = [trial_to_dict(t) for t in disconfirming_trials]
    
    return (
        main_X=main_X, main_y=main_y,
        neutral_X=neutral_X,
        entails_H_X=entails_H_X,
        entails_notH_X=entails_notH_X,
        confirming_X=confirming_X,
        disconfirming_X=disconfirming_X
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Constraint Checking
# ═══════════════════════════════════════════════════════════════════════════════

"""
Compute AUC using Mann-Whitney U statistic.
"""
function compute_auc(scores::Vector{Float64}, labels::Vector{Float64})
    pos_scores = scores[labels .== 1.0]
    neg_scores = scores[labels .== 0.0]
    
    n_pos = length(pos_scores)
    n_neg = length(neg_scores)
    
    if n_pos == 0 || n_neg == 0
        return 0.5
    end
    
    # Count concordant pairs
    u = 0.0
    for ps in pos_scores
        for ns in neg_scores
            if ps > ns
                u += 1.0
            elseif ps == ns
                u += 0.5
            end
        end
    end
    
    return u / (n_pos * n_neg)
end

"""
Check neutrality constraint: measure ≈ 0 when P(H|E) = P(H).
Returns fraction of trials satisfying |measure| < tolerance.
"""
function check_neutrality(tree, evaluate_fn, neutral_X; tolerance=0.05)
    n_satisfied = 0
    for x in neutral_X
        val = evaluate_fn(tree, x)
        if isfinite(val) && abs(val) < tolerance
            n_satisfied += 1
        end
    end
    return n_satisfied / length(neutral_X)
end

"""
Check directionality: measure > 0 when confirming, < 0 when disconfirming.
Returns fraction of trials with correct sign.
"""
function check_directionality(tree, evaluate_fn, confirming_X, disconfirming_X)
    n_correct = 0
    n_total = 0
    
    for x in confirming_X
        val = evaluate_fn(tree, x)
        if isfinite(val)
            n_total += 1
            if val > 0
                n_correct += 1
            end
        end
    end
    
    for x in disconfirming_X
        val = evaluate_fn(tree, x)
        if isfinite(val)
            n_total += 1
            if val < 0
                n_correct += 1
            end
        end
    end
    
    return n_total > 0 ? n_correct / n_total : 0.0
end

"""
Check symmetric logicality: |measure at E⊨H| ≈ |measure at E⊨¬H|.
Returns a score based on how symmetric the extremes are.
"""
function check_symmetric_logicality(tree, evaluate_fn, entails_H_X, entails_notH_X)
    # Get values at E ⊨ H
    vals_H = Float64[]
    for x in entails_H_X
        val = evaluate_fn(tree, x)
        if isfinite(val)
            push!(vals_H, val)
        end
    end
    
    # Get values at E ⊨ ¬H
    vals_notH = Float64[]
    for x in entails_notH_X
        val = evaluate_fn(tree, x)
        if isfinite(val)
            push!(vals_notH, val)
        end
    end
    
    if isempty(vals_H) || isempty(vals_notH)
        return 0.0
    end
    
    # Check that E⊨H gives positive values and E⊨¬H gives negative values
    frac_H_positive = mean(vals_H .> 0)
    frac_notH_negative = mean(vals_notH .< 0)
    
    # Check magnitude symmetry: mean|val at H| ≈ mean|val at ¬H|
    mean_abs_H = mean(abs.(vals_H))
    mean_abs_notH = mean(abs.(vals_notH))
    
    # Symmetry score: 1 if perfectly symmetric, 0 if very asymmetric
    if mean_abs_H + mean_abs_notH > 1e-10
        symmetry = 1 - abs(mean_abs_H - mean_abs_notH) / (mean_abs_H + mean_abs_notH)
    else
        symmetry = 0.0
    end
    
    # Combined score
    return 0.5 * (frac_H_positive + frac_notH_negative) * symmetry
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Search using SymbolicOptimization optimize() API
# ═══════════════════════════════════════════════════════════════════════════════

function run_constrained_search(;
        seed::Int=42,
        population_size::Int=500,
        generations::Int=150,
        n_main::Int=3000,
        min_neutrality::Float64=0.95,      # Require near-perfect (was 0.7)
        min_directionality::Float64=0.95,  # Require near-perfect (was 0.85)
        min_symmetry::Float64=0.90)        # Require high (was 0.5)
    
    println("="^70)
    println("Constrained Confirmation Measure Search")
    println("="^70)
    println()
    println("Parameters:")
    println("  Population: $population_size")
    println("  Generations: $generations")
    println("  Main trials: $n_main")
    println("  Min neutrality: $min_neutrality")
    println("  Min directionality: $min_directionality")
    println("  Min symmetry: $min_symmetry")
    println()
    
    rng = MersenneTwister(seed)
    
    # Create datasets
    println("Generating data...")
    train_data = create_dataset(rng, n_main=n_main)
    test_rng = MersenneTwister(seed + 1000)
    test_data = create_dataset(test_rng, n_main=n_main)
    
    println("  Training: $(length(train_data.main_X)) main trials")
    println("  Testing: $(length(test_data.main_X)) main trials")
    println("  Neutral trials: $(length(train_data.neutral_X))")
    println("  Entails H trials: $(length(train_data.entails_H_X))")
    println("  Entails ¬H trials: $(length(train_data.entails_notH_X))")
    println()
    
    # Variables
    all_vars = [:pH, :pE, :pnotH, :pnotE,
                :pH_and_E, :pH_and_notE, :pnotH_and_E, :pnotH_and_notE,
                :pH_E, :pnotH_E, :pH_notE, :pnotH_notE,
                :pE_H, :pnotE_H, :pE_notH, :pnotE_notH]
    
    # Grammar - use / which the package handles with safe_div internally
    # NO CONSTANTS - set constant_range to nothing to completely disable
    # Include step function for piecewise constructions:
    #   step(cond) * A + (1 - step(cond)) * B  =  "if cond ≥ 0 then A else B"
    # Include log for logarithmic measures (e.g., Good's log-likelihood ratio)
    #   The package automatically uses safe_log internally
    grammar = Grammar(
        binary_operators = [+, -, *, /],
        unary_operators = [step_func, log],
        variables = all_vars,
        constant_range = nothing  # NO CONSTANTS - this actually disables them
    )
    
    # Custom objective 1: Minimize (1 - AUC + constraint penalties)
    function auc_with_constraints(tree, data)
        train = data[:train_data]
        min_neut = data[:min_neutrality]
        min_dir = data[:min_directionality]
        min_sym = data[:min_symmetry]
        g = data[:grammar]
        
        # Evaluate on main data
        scores = Float64[]
        for x in train.main_X
            val = evaluate(tree, x)
            push!(scores, isfinite(val) ? val : 0.0)
        end
        
        # Check for degenerate solutions
        if std(scores) < 1e-10
            return 1.0  # Worst possible
        end
        
        # Compute AUC
        auc = compute_auc(scores, train.main_y)
        
        # Check constraints
        eval_fn = (t, x) -> evaluate(t, x)
        neutrality = check_neutrality(tree, eval_fn, train.neutral_X)
        directionality = check_directionality(tree, eval_fn, 
                                              train.confirming_X, train.disconfirming_X)
        symmetry = check_symmetric_logicality(tree, eval_fn,
                                              train.entails_H_X, train.entails_notH_X)
        
        # Soft constraint penalties - VERY HARSH to effectively require satisfaction
        # Any violation is severely penalized to ensure theoretical validity
        constraint_penalty = 0.0
        if neutrality < min_neut
            constraint_penalty += 1.0 * (min_neut - neutrality)  # Harsh!
        end
        if directionality < min_dir
            constraint_penalty += 1.5 * (min_dir - directionality)  # Harshest - this is essential
        end
        if symmetry < min_sym
            constraint_penalty += 0.8 * (min_sym - symmetry)
        end
        
        # Add small complexity penalty to encourage simpler solutions
        # This helps avoid overfit solutions with similar AUC but more nodes
        complexity_penalty = 0.002 * count_nodes(tree)  # ~0.06 for 30 nodes
        
        return 1.0 - auc + constraint_penalty + complexity_penalty
    end
    
    # Create objectives
    objectives = [
        custom_objective(:auc_penalty, auc_with_constraints; minimize=true),
        complexity_objective()
    ]
    
    # Data dictionary for optimization
    opt_data = Dict{Symbol, Any}(
        :train_data => train_data,
        :grammar => grammar,
        :min_neutrality => min_neutrality,
        :min_directionality => min_directionality,
        :min_symmetry => min_symmetry
    )
    
    # Configuration - larger population and more generations for bigger search space with step
    config = NSGAIIConfig(
        population_size = population_size,
        max_generations = generations,
        tournament_size = 7,
        crossover_prob = 0.7,
        mutation_prob = 0.3,
        max_depth = 8,      # Increased: step(cond)*A + (1-step(cond))*B needs depth
        max_nodes = 40,     # Increased for piecewise structures
        parsimony_tolerance = 0.01,  # Prefer simpler when objectives are close
        verbose = true
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Seed population with known confirmation measures
    # This helps the search explore from good starting points
    # ═══════════════════════════════════════════════════════════════════════════
    
    println("Creating seed formulas...")
    
    # Helper to create variables
    v(name) = Variable(name)
    
    # 1. Difference measure: d = P(H|E) - P(H)
    d_tree = FunctionNode(:-, [v(:pH_E), v(:pH)])
    
    # 2. Christensen's s: P(H|E) - P(H|¬E)
    s_tree = FunctionNode(:-, [v(:pH_E), v(:pH_notE)])
    
    # 3. Good's log-likelihood ratio: log(P(E|H) / P(E|¬H))
    l_tree = FunctionNode(:log, [FunctionNode(:/, [v(:pE_H), v(:pE_notH)])])
    
    # 4. Carnap's relevance quotient: P(H|E) / P(H) - 1
    r_tree = FunctionNode(:-, [FunctionNode(:/, [v(:pH_E), v(:pH)]), 
                               FunctionNode(:/, [v(:pH), v(:pH)])])  # -1 via pH/pH
    
    # 5. Crupi's z measure (piecewise):
    # z = step(pH_E - pH) * ((pH_E - pH) / pnotH) + (1 - step(pH_E - pH)) * ((pH_E - pH) / pH)
    diff_term = FunctionNode(:-, [v(:pH_E), v(:pH)])
    step_diff = FunctionNode(:step_func, [copy_tree(diff_term)])
    z_pos = FunctionNode(:/, [copy_tree(diff_term), v(:pnotH)])
    z_neg = FunctionNode(:/, [copy_tree(diff_term), v(:pH)])
    one_minus_step = FunctionNode(:-, [FunctionNode(:/, [v(:pH), v(:pH)]), copy_tree(step_diff)])  # 1 - step via pH/pH - step
    z_tree = FunctionNode(:+, [
        FunctionNode(:*, [copy_tree(step_diff), z_pos]),
        FunctionNode(:*, [one_minus_step, z_neg])
    ])
    
    # 6. ch measure (piecewise, based on P(H|¬E)):
    # ch = step(pH_E - pH_notE) * ((pH_E - pH_notE) / (1 - pH_notE)) + ...
    diff_s = FunctionNode(:-, [v(:pH_E), v(:pH_notE)])
    step_s = FunctionNode(:step_func, [copy_tree(diff_s)])
    ch_pos = FunctionNode(:/, [copy_tree(diff_s), v(:pnotH_notE)])  # 1 - P(H|¬E) = P(¬H|¬E)
    ch_neg = FunctionNode(:/, [copy_tree(diff_s), v(:pH_notE)])
    one_minus_step_s = FunctionNode(:-, [FunctionNode(:/, [v(:pH), v(:pH)]), copy_tree(step_s)])
    ch_tree = FunctionNode(:+, [
        FunctionNode(:*, [copy_tree(step_s), ch_pos]),
        FunctionNode(:*, [one_minus_step_s, ch_neg])
    ])
    
    # 7. Kemeny-Oppenheim measure: (P(E|H) - P(E|¬H)) / (P(E|H) + P(E|¬H))
    ko_tree = FunctionNode(:/, [
        FunctionNode(:-, [v(:pE_H), v(:pE_notH)]),
        FunctionNode(:+, [v(:pE_H), v(:pE_notH)])
    ])
    
    # 8. Rips measure: (P(H|E) - P(H)) / P(¬H)
    rips_tree = FunctionNode(:/, [FunctionNode(:-, [v(:pH_E), v(:pH)]), v(:pnotH)])
    
    # 9. Plain likelihood ratio (without log): P(E|H) / P(E|¬H)
    lr_tree = FunctionNode(:/, [v(:pE_H), v(:pE_notH)])
    
    # 10. Log posterior odds: log(P(H|E) / P(¬H|E))
    log_post_odds_tree = FunctionNode(:log, [FunctionNode(:/, [v(:pH_E), v(:pnotH_E)])])
    
    # Create Individual objects for seeds
    seed_trees = [d_tree, s_tree, l_tree, r_tree, z_tree, ch_tree, ko_tree, rips_tree, lr_tree, log_post_odds_tree]
    seed_population = [Individual(copy_tree(t)) for t in seed_trees]
    
    # Also add some variants with step functions applied to simple measures
    for base in [d_tree, s_tree]
        # step(base) * base_normalized_pos + (1-step) * base_normalized_neg
        step_base = FunctionNode(:step_func, [copy_tree(base)])
        variant = FunctionNode(:+, [
            FunctionNode(:*, [step_base, FunctionNode(:/, [copy_tree(base), v(:pnotH)])]),
            FunctionNode(:*, [
                FunctionNode(:-, [FunctionNode(:/, [v(:pH), v(:pH)]), copy_tree(step_base)]),
                FunctionNode(:/, [copy_tree(base), v(:pH)])
            ])
        ])
        push!(seed_population, Individual(variant))
    end
    
    println("  Seeded $(length(seed_population)) known measures")
    
    # Run evolution
    println("Running evolution...")
    println()
    
    result = optimize(grammar, objectives, opt_data; 
                      config=config, 
                      rng=MersenneTwister(seed),
                      initial_population=seed_population)
    
    # Get best solution (best for first objective = AUC)
    best_ind = get_best(result, 1)
    best_tree = best_ind.tree
    
    # Evaluate best solution
    println()
    println("="^70)
    println("Best Solution Analysis")
    println("="^70)
    println()
    println("Formula: ", node_to_string(best_tree))
    println()
    
    # Create evaluate function for constraint checking on test set
    eval_fn = (tree, x) -> evaluate(tree, x)
    
    # Test set evaluation
    test_scores = Float64[]
    for x in test_data.main_X
        val = evaluate(best_tree, x)
        push!(test_scores, isfinite(val) ? val : 0.0)
    end
    test_auc = compute_auc(test_scores, test_data.main_y)
    
    # Constraint satisfaction on test set
    test_neutrality = check_neutrality(best_tree, eval_fn, test_data.neutral_X)
    test_directionality = check_directionality(best_tree, eval_fn,
                                                test_data.confirming_X, 
                                                test_data.disconfirming_X)
    test_symmetry = check_symmetric_logicality(best_tree, eval_fn,
                                                test_data.entails_H_X,
                                                test_data.entails_notH_X)
    
    println("Test Set Results:")
    println("  AUC:            $(round(test_auc, digits=4))")
    println("  Neutrality:     $(round(test_neutrality, digits=4)) (min: $min_neutrality)")
    println("  Directionality: $(round(test_directionality, digits=4)) (min: $min_directionality)")
    println("  Symmetry:       $(round(test_symmetry, digits=4)) (min: $min_symmetry)")
    println()
    
    # Compare with z and ch measures
    println("Comparison with existing measures:")
    
    z_scores = Float64[]
    ch_scores = Float64[]
    for x in test_data.main_X
        # z measure
        d = x[:pH_E] - x[:pH]
        z = d >= 0 ? d / x[:pnotH] : d / x[:pH]
        push!(z_scores, isfinite(z) ? z : 0.0)
        
        # ch measure
        s = x[:pH_E] - x[:pH_notE]
        ch = s >= 0 ? s / (1 - x[:pH_notE]) : s / x[:pH_notE]
        push!(ch_scores, isfinite(ch) ? ch : 0.0)
    end
    z_auc = compute_auc(z_scores, test_data.main_y)
    ch_auc = compute_auc(ch_scores, test_data.main_y)
    
    println()
    println("  Measure     │   AUC   │ Neutrality │ Directionality │ Symmetry")
    println("  ────────────┼─────────┼────────────┼────────────────┼──────────")
    @printf("  z           │ %7.4f │   %6.4f   │     %6.4f     │  %6.4f\n", 
            z_auc, 1.0, 1.0, 1.0)  # z satisfies all by construction
    @printf("  ch          │ %7.4f │   %6.4f   │     %6.4f     │  %6.4f\n",
            ch_auc, 1.0, 1.0, 1.0)  # ch satisfies all by construction
    @printf("  Discovered  │ %7.4f │   %6.4f   │     %6.4f     │  %6.4f\n",
            test_auc, test_neutrality, test_directionality, test_symmetry)
    println()
    println("  Δ AUC vs z:  $(round(test_auc - z_auc, digits=4))")
    println("  Δ AUC vs ch: $(round(test_auc - ch_auc, digits=4))")
    
    return (result=result, best_tree=best_tree,
            test_auc=test_auc, test_neutrality=test_neutrality,
            test_directionality=test_directionality, test_symmetry=test_symmetry)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run Search
# ═══════════════════════════════════════════════════════════════════════════════

# Standard run - larger population and more generations for piecewise search
result = run_constrained_search(
    population_size = 1200,
    generations = 300,
    n_main = 3000
)

# Print final formula
println()
println("="^70)
println("FINAL RESULT")
println("="^70)
println()
println("Discovered formula:")
println("  ", node_to_string(result.best_tree))

# Print Pareto front
println()
println("Pareto Front:")
front = get_pareto_front(result.result)
for (i, ind) in enumerate(sort(front, by=x->x.objectives[1])[1:min(5, length(front))])
    expr = node_to_string(ind.tree)
    if length(expr) > 60
        expr = expr[1:57] * "..."
    end
    @printf("  %d. AUC=%.4f, nodes=%d: %s\n", i, 1 - ind.objectives[1], Int(ind.objectives[2]), expr)
end

# Interpretation guide
println()
println("="^70)
println("INTERPRETATION GUIDE")
println("="^70)
println("""

Piecewise functions use step_func(x) = 1 if x ≥ 0, else 0

Pattern: step(cond) * A + (1 - step(cond)) * B
Meaning: "if cond ≥ 0 then A else B"

Example - Crupi's z measure:
  z = step(pH_E - pH) * ((pH_E - pH) / pnotH) + 
      (1 - step(pH_E - pH)) * ((pH_E - pH) / pH)

Which means:
  z = (P(H|E) - P(H)) / (1 - P(H))  if confirming (P(H|E) ≥ P(H))
  z = (P(H|E) - P(H)) / P(H)        if disconfirming (P(H|E) < P(H))
""")