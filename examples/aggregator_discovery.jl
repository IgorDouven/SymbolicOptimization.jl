#=
Aggregator Discovery: Replication of Douven, Kriegeskorte & Stinson (2026)
===========================================================================

This example demonstrates using SymbolicOptimization.jl to discover
probability aggregation formulas, replicating the methodology from:

"Discovering Sources of Crowd Wisdom via Symbolic Search"
(Douven, Kriegeskorte & Stinson, 2026)

Key findings to replicate:
- The dispersion-adaptive means: μ(p^r) where r = Var[ln p]
- These aggregators outperform classical methods (arithmetic, geometric, 
  harmonic, power, Lehmer means) on the Brier score
- They exploit the finding that false statements have higher response 
  variance than true statements

This example uses synthetic data that mimics the statistical properties
of the Stinson et al. (2025) dataset.

Run from package directory:
    julia --project=. examples/aggregator_discovery.jl
=#

using SymbolicOptimization
using SymbolicOptimization: count_nodes  # For diagnostic
using Random
using Statistics
using Printf

# Uncomment to use real data (requires NPZ package):
# using NPZ

println("="^70)
println("Aggregator Discovery - Replication Study")
println("Based on Douven, Kriegeskorte & Stinson (2026)")
println("="^70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate Synthetic Forecasting Data
# ─────────────────────────────────────────────────────────────────────────────

println("\n1. Generating Synthetic Data")
println("-"^50)

"""
Generate synthetic probability forecasts that mimic real forecasting data.

Key properties from Stinson et al. (2025):
- Multiple forecasters rating binary claims
- Variance asymmetry: false statements have higher response variance
- Forecasters are imperfect but positively correlated with truth

Returns: (P, y, train_idx, test_idx) where
- P: n_items × n_forecasters matrix of probability judgments
- y: n_items vector of ground truth (0 or 1)
- train_idx, test_idx: indices for train/test split
"""
function generate_synthetic_forecasts(;
    n_items::Int = 1200,
    n_forecasters::Int = 100,
    base_rate::Float64 = 0.5,
    mean_skill::Float64 = 0.7,      # Average forecaster "skill" (correlation with truth)
    skill_variance::Float64 = 0.15,  # Variation in forecaster skill
    variance_asymmetry::Float64 = 0.4,  # How much more variance for false items
    train_fraction::Float64 = 0.7,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    # Generate ground truth (1 = true statement, 0 = false statement)
    y = Int.(rand(rng, n_items) .< base_rate)
    
    # Generate forecaster skill levels (how well each forecaster tracks truth)
    forecaster_skills = clamp.(mean_skill .+ randn(rng, n_forecasters) .* skill_variance, 0.5, 0.95)
    
    # Generate item difficulties (how hard each item is)
    item_difficulties = rand(rng, n_items) .* 0.3 .+ 0.1  # Range 0.1-0.4
    
    # Generate probability judgments
    P = zeros(n_items, n_forecasters)
    
    for i in 1:n_items
        truth = y[i]
        difficulty = item_difficulties[i]
        
        # False statements have higher base variance (the key asymmetry)
        # This models the finding that people disagree more on falsehoods
        if truth == 1
            base_variance = 0.04
        else
            base_variance = 0.04 * (1 + variance_asymmetry)  # ~40% more variance
        end
        
        for f in 1:n_forecasters
            skill = forecaster_skills[f]
            
            # Base prediction: skilled forecasters are closer to truth
            if truth == 1
                base_pred = 0.5 + skill * 0.4 - difficulty * 0.2
            else
                base_pred = 0.5 - skill * 0.4 + difficulty * 0.2
            end
            
            # Add heterogeneous noise 
            # False items get extra variance from "distance from truth" effect
            noise_sd = sqrt(base_variance) * (1.0 + 0.5 * (1 - skill))
            if truth == 0
                # Extra disagreement on false items (Popper's verisimilitude effect)
                noise_sd *= (1.0 + 0.3 * rand(rng))
            end
            
            pred = base_pred + randn(rng) * noise_sd
            
            # Clamp to valid probability range with small buffer
            P[i, f] = clamp(pred, 0.01, 0.99)
        end
    end
    
    # Train/test split
    perm = shuffle(rng, 1:n_items)
    n_train = round(Int, train_fraction * n_items)
    train_idx = perm[1:n_train]
    test_idx = perm[n_train+1:end]
    
    return P, y, train_idx, test_idx
end

# Generate data
rng = Random.MersenneTwister(42)

# ══════════════════════════════════════════════════════════════════════════════
# OPTION A: Use synthetic data (default)
# ══════════════════════════════════════════════════════════════════════════════
P, y, train_idx, test_idx = generate_synthetic_forecasts(
    n_items = 1200,
    n_forecasters = 100,
    rng = rng
)

# ══════════════════════════════════════════════════════════════════════════════
# OPTION B: Use real Stinson et al. data
# Uncomment this block and comment out OPTION A above
# Requires: using NPZ (add to imports at top)
# ══════════════════════════════════════════════════════════════════════════════
#=
RATINGS_FILE = "/path/to/your/ratings.npy"   # <- Set your path here
TRUTHS_FILE  = "/path/to/your/truths.npy"    # <- Set your path here

P_raw = npzread(RATINGS_FILE)
truths_raw = vec(npzread(TRUTHS_FILE))

# Skip first 60 items (as in original code)
P = Matrix{Float64}(P_raw[61:end, :])
y = Int.(truths_raw[61:end])

n_items = size(P, 1)

# Create train/test split (70/30)
perm = shuffle(rng, 1:n_items)
n_train = round(Int, 0.7 * n_items)
train_idx = perm[1:n_train]
test_idx = perm[n_train+1:end]

println("Loaded real data from Stinson et al.")
=#
# ══════════════════════════════════════════════════════════════════════════════

n_items, n_forecasters = size(P)
println("Generated data:")
println("  Items: $n_items (train: $(length(train_idx)), test: $(length(test_idx)))")
println("  Forecasters: $n_forecasters")
println("  Base rate: $(round(mean(y), digits=3))")

# Verify variance asymmetry
true_items = findall(y .== 1)
false_items = findall(y .== 0)
var_true = mean([var(P[i, :]) for i in true_items])
var_false = mean([var(P[i, :]) for i in false_items])
println("  Variance (true items): $(round(var_true, digits=4))")
println("  Variance (false items): $(round(var_false, digits=4))")
println("  Ratio (false/true): $(round(var_false/var_true, digits=2))")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Define Grammar (Vector Operations)
# ─────────────────────────────────────────────────────────────────────────────

println("\n2. Grammar Definition")
println("-"^50)

#=
The grammar operates on probability vectors, ensuring all discovered formulas
are "participant-neutral" (they cannot reference specific forecasters).

From the paper (Box 2):
- Reductions: vector → scalar (mean, var, sum, min, max, std)
- Element-wise transforms: vector → vector (log, exp, sq, sqrt, inv)
- Broadcast: (vector, scalar) → vector (pow, mul, add)
- Scalar ops: scalar → scalar (add, sub, mul, div, pow, log, exp)
=#

# Define safe operations
const EPS = 1e-8

safe_log_vec(v) = log.(clamp.(v, EPS, Inf))
safe_exp_vec(v) = exp.(clamp.(v, -20.0, 20.0))
safe_sqrt_vec(v) = sqrt.(clamp.(v, 0.0, Inf))
safe_inv_vec(v) = 1.0 ./ clamp.(abs.(v), EPS, Inf)
ew_sq(v) = v .^ 2
ew_flip(v) = 1.0 .- v

# Broadcast operations
ew_pow_s(v, s) = v .^ clamp(s, -10, 10)
ew_mul_s(v, s) = v .* s
ew_add_s(v, s) = v .+ s

# Scalar operations
safe_s_log(x) = log(clamp(abs(x), EPS, Inf))
safe_s_exp(x) = exp(clamp(x, -20, 20))
safe_s_inv(x) = 1.0 / clamp(abs(x), EPS, Inf)
safe_s_pow(x, y) = abs(x)^clamp(y, -10, 10)
safe_s_div(x, y) = x / clamp(abs(y), EPS, Inf)
s_neg(x) = -x

# Output normalization
clamp01(x) = clamp(x, 0.0, 1.0)
my_sigmoid(x) = 1.0 / (1.0 + exp(-clamp(x, -20, 20)))

grammar = Grammar(
    types = [:Scalar, :Vector],
    
    variables = [
        :ps => :Vector,  # Probability vector (the input)
    ],
    
    operators = [
        # === Reductions: Vector → Scalar ===
        (sum, [:Vector] => :Scalar),
        (mean, [:Vector] => :Scalar),
        (maximum, [:Vector] => :Scalar),
        (minimum, [:Vector] => :Scalar),
        (std, [:Vector] => :Scalar),
        (var, [:Vector] => :Scalar),
        
        # === Element-wise unary: Vector → Vector ===
        (safe_log_vec, [:Vector] => :Vector),
        (safe_exp_vec, [:Vector] => :Vector),
        (ew_sq, [:Vector] => :Vector),
        (safe_sqrt_vec, [:Vector] => :Vector),
        (safe_inv_vec, [:Vector] => :Vector),
        (ew_flip, [:Vector] => :Vector),
        
        # === Broadcast: (Vector, Scalar) → Vector ===
        (ew_pow_s, [:Vector, :Scalar] => :Vector),
        (ew_mul_s, [:Vector, :Scalar] => :Vector),
        (ew_add_s, [:Vector, :Scalar] => :Vector),
        
        # === Scalar operations ===
        (+, [:Scalar, :Scalar] => :Scalar),
        (-, [:Scalar, :Scalar] => :Scalar),
        (*, [:Scalar, :Scalar] => :Scalar),
        (safe_s_div, [:Scalar, :Scalar] => :Scalar),
        (safe_s_pow, [:Scalar, :Scalar] => :Scalar),
        (safe_s_log, [:Scalar] => :Scalar),
        (safe_s_exp, [:Scalar] => :Scalar),
        (safe_s_inv, [:Scalar] => :Scalar),
        (s_neg, [:Scalar] => :Scalar),
        
        # === Output normalization ===
        (clamp01, [:Scalar] => :Scalar),
        (my_sigmoid, [:Scalar] => :Scalar),
    ],
    
    constant_types = [:Scalar],
    constant_range = (-3.0, 3.0),
    output_type = :Scalar,
)

println("Grammar defined with $(length(grammar.operators)) operators")
println("Types: Vector, Scalar")
println("Variable: ps (probability vector)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Define Objectives
# ─────────────────────────────────────────────────────────────────────────────

println("\n3. Objective Functions")
println("-"^50)

#=
Three objectives (as in the paper):
1. Brier score: (prediction - outcome)² averaged over items
2. AUC loss: 1 - AUC (area under ROC curve)
3. Complexity: number of nodes in the expression tree
=#

"""
Compute predictions for all items using a given aggregator tree.
"""
function compute_predictions(tree::AbstractNode, P_eval::Matrix{Float64}, grammar)
    n_items = size(P_eval, 1)
    preds = zeros(n_items)
    
    for i in 1:n_items
        ps = P_eval[i, :]
        ctx = EvalContext(ps=ps)
        pred = try
            evaluate(tree, grammar, ctx)
        catch
            NaN
        end
        
        # Handle invalid outputs
        if pred isa AbstractVector
            preds[i] = NaN
        elseif isnan(pred) || !isfinite(pred)
            preds[i] = NaN
        else
            preds[i] = clamp(pred, 0.0, 1.0)
        end
    end
    
    return preds
end

"""
Compute AUC (Area Under ROC Curve) for binary classification.
"""
function compute_auc(preds::Vector{Float64}, y_eval::Vector{Int})
    valid = .!isnan.(preds)
    if sum(valid) < 10
        return 0.5
    end
    
    preds_v = preds[valid]
    y_v = y_eval[valid]
    
    pos_idx = findall(y_v .== 1)
    neg_idx = findall(y_v .== 0)
    
    n_pos, n_neg = length(pos_idx), length(neg_idx)
    (n_pos == 0 || n_neg == 0) && return 0.5
    
    wins = 0.0
    ties = 0.0
    for i in pos_idx
        pi = preds_v[i]
        for j in neg_idx
            pj = preds_v[j]
            if pi > pj
                wins += 1
            elseif pi == pj
                ties += 1
            end
        end
    end
    
    return (wins + 0.5 * ties) / (n_pos * n_neg)
end

"""
Brier score objective: mean squared error between predictions and outcomes.
"""
function brier_objective_func(tree::AbstractNode, data::Dict)
    P_data = data[:P]
    y_data = data[:y]
    idx = data[:eval_idx]
    grammar_data = data[:grammar]
    
    P_eval = P_data[idx, :]
    y_eval = y_data[idx]
    
    preds = compute_predictions(tree, P_eval, grammar_data)
    
    valid = .!isnan.(preds)
    if sum(valid) < length(preds) * 0.5
        return Inf  # Too many invalid predictions
    end
    
    brier = mean((preds[valid] .- y_eval[valid]).^2)
    return brier
end

"""
AUC loss objective: 1 - AUC (so we minimize).
"""
function auc_loss_objective_func(tree::AbstractNode, data::Dict)
    P_data = data[:P]
    y_data = data[:y]
    idx = data[:eval_idx]
    grammar_data = data[:grammar]
    
    P_eval = P_data[idx, :]
    y_eval = y_data[idx]
    
    preds = compute_predictions(tree, P_eval, grammar_data)
    auc = compute_auc(preds, y_eval)
    
    return 1.0 - auc
end

brier_obj = custom_objective(:brier, brier_objective_func; minimize=true)
auc_loss_obj = custom_objective(:auc_loss, auc_loss_objective_func; minimize=true)
complexity_obj = complexity_objective()

objectives = [brier_obj, auc_loss_obj, complexity_obj]
println("Objectives: Brier score, AUC loss (1-AUC), Complexity")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build Classical Aggregator Seeds
# ─────────────────────────────────────────────────────────────────────────────

println("\n4. Building Classical Aggregator Seeds")
println("-"^50)

"""
Build initial population seeded with classical aggregation methods.
This ensures known good solutions are considered in the search.
"""
function build_classical_seeds()
    seeds = Individual[]
    
    # Helper to create trees
    ps = Variable(:ps, :Vector)
    
    # 1. Arithmetic mean: clamp01(mean(ps))
    arith = FunctionNode(:clamp01, 
        FunctionNode(:mean, ps))
    push!(seeds, Individual(arith))
    
    # 2. Geometric mean: clamp01(exp(mean(log(ps))))
    geom = FunctionNode(:clamp01,
        FunctionNode(:safe_s_exp,
            FunctionNode(:mean,
                FunctionNode(:safe_log_vec, ps))))
    push!(seeds, Individual(geom))
    
    # 3. Harmonic mean: clamp01(1 / mean(1/ps))
    harm = FunctionNode(:clamp01,
        FunctionNode(:safe_s_inv,
            FunctionNode(:mean,
                FunctionNode(:safe_inv_vec, ps))))
    push!(seeds, Individual(harm))
    
    # 4. Power means M_r for various r: (mean(ps^r))^(1/r)
    for r in [-2.0, -1.0, -0.5, 0.5, 2.0, 3.0]
        power_mean = FunctionNode(:clamp01,
            FunctionNode(:safe_s_pow,
                FunctionNode(:mean,
                    FunctionNode(:ew_pow_s, ps, Constant(r))),
                Constant(1.0 / r)))
        push!(seeds, Individual(power_mean))
    end
    
    # 5. Lehmer means L_r: sum(ps^r) / sum(ps^(r-1))
    for r in [0.5, 1.5, 2.0, 2.5]
        lehmer = FunctionNode(:clamp01,
            FunctionNode(:safe_s_div,
                FunctionNode(:sum,
                    FunctionNode(:ew_pow_s, ps, Constant(r))),
                FunctionNode(:sum,
                    FunctionNode(:ew_pow_s, ps, Constant(r - 1.0)))))
        push!(seeds, Individual(lehmer))
    end
    
    # 6. Quadratic mean: sqrt(mean(ps²))
    quad = FunctionNode(:clamp01,
        FunctionNode(:safe_s_pow,
            FunctionNode(:mean,
                FunctionNode(:ew_sq, ps)),
            Constant(0.5)))
    push!(seeds, Individual(quad))
    
    # NOTE: We intentionally do NOT seed the adaptive power mean μ(ps^(Var[ln ps]))
    # The goal is to discover it through search, replicating the paper's methodology.
    
    # 7. Building block: mean raised to a power (helps find μ(p)^r structure)
    # This seeds the idea of raising the mean to a computed exponent
    mean_to_std = FunctionNode(:clamp01,
        FunctionNode(:safe_s_pow,
            FunctionNode(:mean, ps),
            FunctionNode(:std, ps)))  # mean(ps)^std(ps)
    push!(seeds, Individual(mean_to_std))
    
    # 8. Building block: power mean with computed exponent
    # Uses variance (not of log) as exponent - close but not the target
    pow_by_var = FunctionNode(:clamp01,
        FunctionNode(:mean,
            FunctionNode(:ew_pow_s, ps, 
                FunctionNode(:var, ps))))  # mean(ps^var(ps))
    push!(seeds, Individual(pow_by_var))
    
    return seeds
end

seeds = build_classical_seeds()
println("Created $(length(seeds)) classical aggregator seeds")

# Diagnostic: evaluate seeds directly to verify they work (only if using seeds)
# This will be shown regardless, but only relevant when USE_SEEDS=true
println("\nDiagnostic: Evaluating seeds on training data...")
train_data = Dict{Symbol, Any}(
    :P => P,
    :y => y,
    :eval_idx => train_idx,
    :grammar => grammar,
)
for (i, seed) in enumerate(seeds)
    brier = brier_objective_func(seed.tree, train_data)
    auc_loss = auc_loss_objective_func(seed.tree, train_data)
    nodes = count_nodes(seed.tree)
    expr = node_to_string(seed.tree)
    if length(expr) > 50
        expr = expr[1:47] * "..."
    end
    @printf("  Seed %2d: Brier=%.4f  AUC_loss=%.4f  nodes=%2d  %s\n", 
            i, brier, auc_loss, nodes, expr)
end
println()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Run Optimization
# ─────────────────────────────────────────────────────────────────────────────

println("\n5. Running NSGA-II Optimization")
println("-"^50)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION: Toggle seeding on/off
# ══════════════════════════════════════════════════════════════════════════════
USE_SEEDS = false  # Set to false for pure discovery from random initialization
# ══════════════════════════════════════════════════════════════════════════════

if USE_SEEDS
    # Seeded run: smaller population, fewer generations
    config = NSGAIIConfig(
        population_size = 500,
        max_generations = 300,
        max_depth = 5,
        max_nodes = 12,              # Tighter limit
        tournament_size = 3,
        crossover_prob = 0.7,
        mutation_prob = 0.8,
        elite_fraction = 0.02,
        parsimony_tolerance = 0.05,  # 5% tolerance: prefer simpler when Brier within 5%
        simplify_prob = 0.15,        # Higher simplification
        early_stop_generations = 100,
        verbose = true,
    )
    initial_pop = seeds
    println("Mode: SEEDED ($(length(seeds)) classical aggregators)")
else
    # Unseeded run: larger population, more generations for pure discovery
    config = NSGAIIConfig(
        population_size = 1000,     # Larger population for diversity
        max_generations = 500,      # More generations to explore
        max_depth = 5,
        max_nodes = 12,             # Tighter limit to prevent bloat
        tournament_size = 2,        # Smaller tournament = less selection pressure = more exploration
        crossover_prob = 0.7,
        mutation_prob = 0.9,        # Higher mutation for exploration
        elite_fraction = 0.01,      # Minimal elitism to maintain diversity
        parsimony_tolerance = 0.05, # 5% tolerance: prefer simpler when Brier within 5%
        simplify_prob = 0.2,        # Higher simplification to combat bloat
        early_stop_generations = 150,
        verbose = true,
    )
    initial_pop = nothing
    println("Mode: UNSEEDED (pure discovery from random initialization)")
end

# Prepare data dict
data = Dict{Symbol, Any}(
    :P => P,
    :y => y,
    :eval_idx => train_idx,
    :grammar => grammar,
)

println("Configuration:")
println("  Population: $(config.population_size)")
println("  Generations: $(config.max_generations)")
println("  Max depth: $(config.max_depth)")
println("  Max nodes: $(config.max_nodes)")
println("  Parsimony tolerance: $(config.parsimony_tolerance > 0 ? "$(Int(config.parsimony_tolerance * 100))%" : "disabled")")

result = optimize(grammar, objectives, data;
    config = config,
    rng = rng,
    initial_population = initial_pop
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluate Results on Test Set
# ─────────────────────────────────────────────────────────────────────────────

println("\n6. Test Set Evaluation")
println("-"^50)

# Update data for test evaluation
test_data = Dict{Symbol, Any}(
    :P => P,
    :y => y,
    :eval_idx => test_idx,
    :grammar => grammar,
)

# Get Pareto front
front = get_pareto_front(result)
println("Pareto front size: $(length(front))")

# Evaluate all front members on test set
test_results = []
for ind in front
    test_brier = brier_objective_func(ind.tree, test_data)
    test_auc_loss = auc_loss_objective_func(ind.tree, test_data)
    nodes = Int(ind.objectives[3])
    push!(test_results, (
        ind = ind,
        train_brier = ind.objectives[1],
        train_auc_loss = ind.objectives[2],
        complexity = nodes,
        test_brier = test_brier,
        test_auc = 1.0 - test_auc_loss,
    ))
end

# Filter out invalid results (Inf, NaN)
valid_results = filter(r -> isfinite(r.test_brier), test_results)
println("Valid results: $(length(valid_results)) / $(length(test_results))")

if isempty(valid_results)
    println("\n⚠ WARNING: No valid results found! All solutions produced Inf/NaN on test set.")
    println("This may indicate a problem with the grammar or evaluation.")
    # Fall back to test_results for reporting, but warn user
    valid_results = test_results
end

# Sort by test Brier (for reference)
sort!(valid_results, by = r -> isfinite(r.test_brier) ? r.test_brier : Inf)

# Parsimony-aware sorting: group by tolerance, then prefer simpler within group
PARSIMONY_TOLERANCE = config.parsimony_tolerance  # Use same tolerance as during evolution
function parsimony_sort_key(r)
    # Handle Inf/NaN - put them last
    if !isfinite(r.test_brier)
        return (Inf, r.complexity, Inf)
    end
    # Quantize Brier to tolerance buckets, then use complexity as tiebreaker
    bucket = floor(r.test_brier / PARSIMONY_TOLERANCE)
    return (bucket, r.complexity, r.test_brier)
end
test_results_parsimony = sort(valid_results, by = parsimony_sort_key)

# Also get unique formulas (deduplicate)
seen_formulas = Set{String}()
unique_results = []
for r in test_results_parsimony
    formula = node_to_string(r.ind.tree)
    if formula ∉ seen_formulas
        push!(seen_formulas, formula)
        push!(unique_results, r)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Compare with Classical Baselines
# ─────────────────────────────────────────────────────────────────────────────

println("\n7. Comparison with Baselines")
println("-"^50)

P_test = P[test_idx, :]
y_test = y[test_idx]

function eval_baseline(name, preds)
    valid = .!isnan.(preds)
    brier = mean((preds[valid] .- y_test[valid]).^2)
    auc = compute_auc(preds, y_test)
    acc = mean((preds .>= 0.5) .== (y_test .== 1))
    @printf("  %-25s Brier=%.4f  AUC=%.4f  Acc=%.3f\n", name, brier, auc, acc)
    return brier
end

println("\nClassical Aggregators:")
eval_baseline("Arithmetic mean", vec(mean(P_test, dims=2)))
eval_baseline("Geometric mean", vec(exp.(mean(log.(clamp.(P_test, EPS, 1-EPS)), dims=2))))
eval_baseline("Harmonic mean", vec(1.0 ./ mean(1.0 ./ clamp.(P_test, EPS, 1), dims=2)))
eval_baseline("Quadratic mean (M₂)", vec(sqrt.(mean(P_test.^2, dims=2))))

for r in [0.5, -1.0, 3.0]
    preds = vec(mean(P_test.^r, dims=2).^(1/r))
    eval_baseline("Power mean M_$r", preds)
end

# Adaptive power mean (the target)
println("\nDispersion-Adaptive Aggregators:")
p_adaptive = zeros(size(P_test, 1))
for i in axes(P_test, 1)
    ps = P_test[i, :]
    r = var(log.(clamp.(ps, EPS, 1-EPS)))
    p_adaptive[i] = mean(ps.^r)
end
adaptive_brier = eval_baseline("Adaptive μ(p^r), r=Var[ln p]", p_adaptive)

# Mean raised to adaptive power
p_mean_pow = zeros(size(P_test, 1))
for i in axes(P_test, 1)
    ps = P_test[i, :]
    r = var(log.(clamp.(ps, EPS, 1-EPS)))
    p_mean_pow[i] = mean(ps)^r
end
eval_baseline("μ(p)^r, r=Var[ln p]", p_mean_pow)

# Adaptive squared
p_adaptive_sq = zeros(size(P_test, 1))
for i in axes(P_test, 1)
    ps = P_test[i, :]
    r = var(log.(clamp.(ps, EPS, 1-EPS)))
    p_adaptive_sq[i] = mean(ps.^(r^2))
end
eval_baseline("Adaptive μ(p^r²)", p_adaptive_sq)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Report Discovered Aggregators
# ─────────────────────────────────────────────────────────────────────────────

println("\n8. Top Discovered Aggregators")
println("-"^50)

# Show unique formulas with parsimony-aware ranking
println("\nTop 10 unique formulas (parsimony-aware: simpler preferred within $(Int(PARSIMONY_TOLERANCE*100))% Brier):")
for (i, r) in enumerate(unique_results[1:min(10, length(unique_results))])
    expr = node_to_string(r.ind.tree)
    if length(expr) > 60
        expr = expr[1:57] * "..."
    end
    @printf("[%2d] Brier=%.4f  AUC=%.4f  nodes=%d\n", i, r.test_brier, r.test_auc, r.complexity)
    println("     $expr")
end

# Find simplest competitive solutions (within parsimony tolerance of best Brier)
best_brier = valid_results[1].test_brier
competitive_threshold = best_brier * (1.0 + PARSIMONY_TOLERANCE)
simple_competitive = filter(r -> r.test_brier <= competitive_threshold, unique_results)
sort!(simple_competitive, by = r -> (r.complexity, r.test_brier))

println("\nSimplest competitive solutions (within $(Int(PARSIMONY_TOLERANCE*100))% of best Brier=$(round(best_brier, digits=4))):")
for (i, r) in enumerate(simple_competitive[1:min(10, length(simple_competitive))])
    expr = node_to_string(r.ind.tree)
    if length(expr) > 55
        expr = expr[1:52] * "..."
    end
    @printf("[%2d] nodes=%2d  Brier=%.4f  AUC=%.4f\n", i, r.complexity, r.test_brier, r.test_auc)
    println("     $expr")
end

println("\n" * "="^70)
println("Summary")
println("="^70)

# Best by raw Brier
best_discovered = valid_results[1]
println("\nBest by Brier score:")
println("  Formula: $(node_to_string(best_discovered.ind.tree))")
println("  Test Brier: $(round(best_discovered.test_brier, digits=4))")
println("  Test AUC: $(round(best_discovered.test_auc, digits=4))")
println("  Complexity: $(best_discovered.complexity) nodes")

# Best simple solution (parsimony-aware)
best_simple = unique_results[1]
if best_simple.complexity < best_discovered.complexity
    println("\nBest simple solution (parsimony-aware):")
    println("  Formula: $(node_to_string(best_simple.ind.tree))")
    println("  Test Brier: $(round(best_simple.test_brier, digits=4))")
    println("  Test AUC: $(round(best_simple.test_auc, digits=4))")
    println("  Complexity: $(best_simple.complexity) nodes")
end

println("\nTarget (Adaptive power mean μ(p^r)):")
println("  Test Brier: $(round(adaptive_brier, digits=4))")

# Check if we found a simple solution close to target
simple_near_target = filter(r -> r.test_brier <= adaptive_brier * (1.0 + PARSIMONY_TOLERANCE) && r.complexity <= 8, unique_results)
if !isempty(simple_near_target)
    best_simple_near = simple_near_target[1]
    println("\n✓ Found simple solution (≤8 nodes) within $(Int(PARSIMONY_TOLERANCE*100))% of target performance!")
    println("  Formula: $(node_to_string(best_simple_near.ind.tree))")
    println("  Brier: $(round(best_simple_near.test_brier, digits=4)) vs target $(round(adaptive_brier, digits=4))")
elseif best_discovered.test_brier <= adaptive_brier * 1.05
    println("\n✓ Discovered aggregator matching or exceeding target performance")
    println("  (but solution is complex: $(best_discovered.complexity) nodes)")
else
    pct_worse = (best_discovered.test_brier/adaptive_brier - 1)*100
    if isfinite(pct_worse)
        println("\n→ Best discovered is $(round(pct_worse, digits=1))% worse than target")
    else
        println("\n→ Best discovered has invalid Brier score (Inf)")
    end
    println("  (May need more generations or larger population)")
end

println("\n" * "="^70)
println("Example complete!")
println("="^70)
