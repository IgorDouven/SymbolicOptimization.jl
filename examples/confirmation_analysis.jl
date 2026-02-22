#=
Confirmation Measures: Detailed Analysis
========================================

Follow-up analysis based on initial discovery results:
1. Targeted search seeded with p(H|E) + p(H) - p(E|¬H)
2. Thorough testing of discovered formulas
3. Theoretical analysis: prior-dependent vs prior-independent measures

Run from package directory:
    julia --project=. examples/confirmation_analysis.jl
=#

using SymbolicOptimization
using Random
using Distributions
using Statistics

println("="^70)
println("Confirmation Measures: Detailed Analysis")
println("="^70)

# ═══════════════════════════════════════════════════════════════════════════════
# Setup: Reuse trial generation from main example
# ═══════════════════════════════════════════════════════════════════════════════

function generate_confirmation_trial(rng; n_worlds::Int=20)
    π = rand(rng, Dirichlet(n_worlds, 1.0))
    actual_world = rand(rng, 1:n_worlds)
    
    H_mask = rand(rng, Bool, n_worlds)
    E_mask = rand(rng, Bool, n_worlds)
    E_mask[actual_world] = true
    
    if sum(H_mask) == 0
        H_mask[rand(rng, 1:n_worlds)] = true
    elseif sum(H_mask) == n_worlds
        H_mask[rand(rng, 1:n_worlds)] = false
    end
    if sum(E_mask) == 0
        E_mask[actual_world] = true
    elseif sum(E_mask) == n_worlds
        idx = rand(rng, setdiff(1:n_worlds, [actual_world]))
        E_mask[idx] = false
    end
    
    pH = sum(π[H_mask])
    pE = sum(π[E_mask])
    pnotH = 1 - pH
    pnotE = 1 - pE
    
    pH_and_E = sum(π[H_mask .& E_mask])
    pH_and_notE = sum(π[H_mask .& .!E_mask])
    pnotH_and_E = sum(π[.!H_mask .& E_mask])
    pnotH_and_notE = sum(π[.!H_mask .& .!E_mask])
    
    ε = 1e-10
    pH_E = pH_and_E / max(pE, ε)
    pE_H = pH_and_E / max(pH, ε)
    pH_notE = pH_and_notE / max(pnotE, ε)
    pE_notH = pnotH_and_E / max(pnotH, ε)
    pnotH_E = pnotH_and_E / max(pE, ε)
    pnotE_H = pH_and_notE / max(pH, ε)
    pnotH_notE = pnotH_and_notE / max(pnotE, ε)
    pnotE_notH = pnotH_and_notE / max(pnotH, ε)
    
    inputs = Dict{Symbol, Float64}(
        :pH => pH, :pE => pE, :pnotH => pnotH, :pnotE => pnotE,
        :pH_E => pH_E, :pE_H => pE_H, :pH_notE => pH_notE, :pE_notH => pE_notH,
        :pnotH_E => pnotH_E, :pnotE_H => pnotE_H, 
        :pnotH_notE => pnotH_notE, :pnotE_notH => pnotE_notH,
        :pH_and_E => pH_and_E,
    )
    
    H_is_true = H_mask[actual_world]
    return (inputs, H_is_true)
end

# Evaluation function with confidence intervals
function evaluate_measure_detailed(measure_fn, seed; 
        world_sizes=5:5:100, trials_per_size=500, n_bootstrap=100)
    
    rng = MersenneTwister(seed)
    
    # Collect all scores and labels
    all_scores = Float64[]
    all_labels = Bool[]
    results_by_size = Dict{Int, @NamedTuple{scores::Vector{Float64}, labels::Vector{Bool}}}()
    
    for n_worlds in world_sizes
        scores = Float64[]
        labels = Bool[]
        
        for _ in 1:trials_per_size
            inputs, label = generate_confirmation_trial(rng, n_worlds=n_worlds)
            score = measure_fn(inputs)
            if isfinite(score)
                push!(scores, score)
                push!(labels, label)
                push!(all_scores, score)
                push!(all_labels, label)
            end
        end
        
        results_by_size[n_worlds] = (scores=scores, labels=labels)
    end
    
    # Compute overall AUC
    overall_auc = compute_auc(all_scores, all_labels)
    
    # Bootstrap confidence interval
    bootstrap_aucs = Float64[]
    n = length(all_scores)
    for _ in 1:n_bootstrap
        idx = rand(rng, 1:n, n)
        boot_auc = compute_auc(all_scores[idx], all_labels[idx])
        push!(bootstrap_aucs, boot_auc)
    end
    ci_low = quantile(bootstrap_aucs, 0.025)
    ci_high = quantile(bootstrap_aucs, 0.975)
    
    # AUC by world size
    auc_by_size = Dict(
        nw => compute_auc(r.scores, r.labels) 
        for (nw, r) in results_by_size
    )
    
    # Valid prediction rate
    valid_rate = length(all_scores) / (length(world_sizes) * trials_per_size)
    
    return (
        overall_auc = overall_auc,
        ci_low = ci_low,
        ci_high = ci_high,
        auc_by_size = auc_by_size,
        valid_rate = valid_rate
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Targeted Search Seeded with Best Formula
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("PART 1: Targeted Search")
println("="^70)

println("\nGenerating training data...")
train_rng = MersenneTwister(12345)
world_sizes = 5:5:100
trials_per_size = 100

train_trials = Tuple{Dict{Symbol,Float64}, Bool}[]
for n_worlds in world_sizes
    for _ in 1:trials_per_size
        push!(train_trials, generate_confirmation_trial(train_rng, n_worlds=n_worlds))
    end
end
train_inputs = [t[1] for t in train_trials]
train_labels = [t[2] for t in train_trials]

println("  Training set: $(length(train_trials)) trials")

# Custom evaluator using fixed trials
function fixed_trials_evaluator(; inputs, labels, objectives=[:auc, :complexity])
    return (tree, env, evaluate_fn, count_nodes_fn) -> begin
        var_names = env[:var_names]
        scores = Float64[]
        valid_labels = Bool[]
        
        for (inp, label) in zip(inputs, labels)
            ctx = Dict{Symbol, Any}(name => inp[name] for name in keys(inp))
            score = evaluate_fn(tree, ctx)
            if isfinite(score)
                push!(scores, score)
                push!(valid_labels, label)
            end
        end
        
        obj_values = Float64[]
        for obj in objectives
            if obj == :auc
                auc = length(scores) >= 10 ? compute_auc(scores, valid_labels) : 0.5
                push!(obj_values, 1.0 - auc)
            elseif obj == :complexity
                push!(obj_values, Float64(count_nodes_fn(tree)))
            end
        end
        return obj_values
    end
end

println("\nRunning targeted search seeded with promising formulas...")

result_targeted = solve(policy_problem(
    variables = [:pH, :pE, :pnotH, :pnotE,
                 :pH_E, :pE_H, :pH_notE, :pE_notH,
                 :pnotH_E, :pnotE_H, :pnotH_notE, :pnotE_notH,
                 :pH_and_E],
    
    binary_operators = [+, -, *, /],
    unary_operators = Function[],
    constants = (0.0, 2.0),
    constant_prob = 0.15,
    
    evaluator = fixed_trials_evaluator(
        inputs = train_inputs,
        labels = train_labels,
        objectives = [:auc, :complexity]
    ),
    n_objectives = 2,
    
    # Seed with the best-performing formulas
    seed_formulas = [
        # The winner from custom tests
        v -> v.pH_E + v.pH - v.pE_notH,
        
        # Variations to explore
        v -> v.pH_E + v.pH - v.pnotH_E,
        v -> v.pH_E + v.pE_H - v.pE_notH,
        v -> v.pH_E * v.pH / v.pE_notH,
        v -> (v.pH_E + v.pH) / (1 + v.pE_notH),
        
        # Discovered formula from previous run
        v -> v.pH_E / (v.pnotH + v.pE_notH),
        
        # Classic measures for comparison
        v -> v.pH_E - v.pH,                  # difference (d)
        v -> v.pE_H - v.pE_notH,             # Nozick (n)
        v -> v.pH_E - v.pH_notE,             # Christensen (s)
    ],
    
    population = 300,
    generations = 100,
    max_depth = 5,
    max_nodes = 12,
    seed = 777,
    verbose = true
))

println("\n\nTargeted Search Results:")
println("-"^50)

best_targeted = best(result_targeted)
println("Best formula: $(best_targeted.expression)")
println("Training AUC: $(round(1 - best_targeted.objectives[1], digits=4))")
println("Complexity: $(Int(best_targeted.objectives[2])) nodes")

# Show Pareto front
println("\nPareto Frontier:")
front = pareto_front(result_targeted)
seen = Set{String}()
for sol in sort(front, by=s->s.objectives[1])[1:min(8, length(front))]
    if sol.expression ∉ seen
        push!(seen, sol.expression)
        expr = length(sol.expression) > 50 ? sol.expression[1:47] * "..." : sol.expression
        println("  $(round(1-sol.objectives[1], digits=4)) AUC, $(Int(sol.objectives[2])) nodes: $expr")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Thorough Testing of Key Formulas
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("PART 2: Thorough Testing of Key Formulas")
println("="^70)

# Define all formulas to test
formulas = Dict{String, Function}(
    # New discoveries
    "p(H|E) + p(H) - p(E|¬H)" => i -> i[:pH_E] + i[:pH] - i[:pE_notH],
    "p(H|E) - p(E|¬H)" => i -> i[:pH_E] - i[:pE_notH],
    "p(H|E) / p(E|¬H)" => i -> i[:pH_E] / max(i[:pE_notH], 1e-10),
    "p(H|E) / (p(¬H) + p(E|¬H))" => i -> i[:pH_E] / max(i[:pnotH] + i[:pE_notH], 1e-10),
    
    # Classic measures
    "d: p(H|E) - p(H)" => i -> i[:pH_E] - i[:pH],
    "s: p(H|E) - p(H|¬E)" => i -> i[:pH_E] - i[:pH_notE],
    "n: p(E|H) - p(E|¬H)" => i -> i[:pE_H] - i[:pE_notH],
    "k: (p(E|H)-p(E|¬H))/(p(E|H)+p(E|¬H))" => i -> begin
        num = i[:pE_H] - i[:pE_notH]
        den = i[:pE_H] + i[:pE_notH]
        num / max(den, 1e-10)
    end,
    "z: Crupi et al." => i -> begin
        s = i[:pH_E] - i[:pH_notE]
        if s >= 0
            s / max(i[:pnotH] / max(i[:pnotE], 1e-10), 1e-10)
        else
            s / max(i[:pH] / max(i[:pnotE], 1e-10), 1e-10)
        end
    end,
    "ch: Cheng" => i -> begin
        s = i[:pH_E] - i[:pH_notE]
        if s >= 0
            s / max(i[:pnotH_notE], 1e-10)
        else
            s / max(i[:pH_notE], 1e-10)
        end
    end,
)

println("\nEvaluating formulas with bootstrap confidence intervals...")
println("(10,000 test trials, 100 bootstrap samples)\n")

results = Dict{String, Any}()
for (name, formula) in formulas
    result = evaluate_measure_detailed(formula, 99999)
    results[name] = result
end

# Sort by overall AUC
sorted_results = sort(collect(results), by=x->x[2].overall_auc, rev=true)

println("Formula                              │   AUC   │   95% CI      │ Valid%")
println("─────────────────────────────────────┼─────────┼───────────────┼────────")
for (name, r) in sorted_results
    name_short = length(name) > 36 ? name[1:33] * "..." : name
    auc = round(r.overall_auc, digits=4)
    ci = "[$(round(r.ci_low, digits=3)), $(round(r.ci_high, digits=3))]"
    valid = round(100*r.valid_rate, digits=1)
    println("$(rpad(name_short, 36)) │ $(lpad(auc, 7)) │ $(lpad(ci, 13)) │ $(lpad(valid, 5))%")
end

# Detailed comparison by world size
println("\n\nAUC by Number of Worlds (top 5 formulas):")
println("-"^70)

top5 = sorted_results[1:min(5, length(sorted_results))]
top5_names = [t[1][1:min(20, length(t[1]))] for t in top5]

print("Worlds │")
for name in top5_names
    print(" $(rpad(name, 12)) │")
end
println()
println("───────┼" * repeat("──────────────┼", length(top5_names)))

for nw in [5, 10, 20, 50, 75, 100]
    print("$(lpad(nw, 6)) │")
    for (name, r) in top5
        auc = round(get(r.auc_by_size, nw, NaN), digits=3)
        print(" $(lpad(auc, 12)) │")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: Theoretical Analysis - Prior-Dependent vs Prior-Independent
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("PART 3: Theoretical Analysis")
println("="^70)

println("""

QUESTION: Why do prior-dependent formulas outperform prior-independent ones?

The key insight is that the TASK matters:
- Confirmation theory asks: "How much does E support H?" (evidential impact)
- Truth tracking asks: "Given E, how likely is H true?" (prediction)

For CONFIRMATION (evidential impact):
  Prior-independent measures like z, ch, k make sense because we want to
  isolate what the EVIDENCE tells us, separate from our prior beliefs.

For TRUTH TRACKING (prediction):
  Including the prior is OPTIMAL because p(H) carries genuine information
  about whether H is true in the simulation setup.

""")

# Demonstrate the difference
println("Empirical demonstration:")
println("-"^50)

# Compare prior-dependent vs prior-independent measures
prior_dependent = [
    ("p(H|E) + p(H) - p(E|¬H)", i -> i[:pH_E] + i[:pH] - i[:pE_notH]),
    ("p(H|E)", i -> i[:pH_E]),
]

prior_independent = [
    ("d: p(H|E) - p(H)", i -> i[:pH_E] - i[:pH]),
    ("z: Crupi et al.", i -> begin
        s = i[:pH_E] - i[:pH_notE]
        if s >= 0
            s / max(i[:pnotH] / max(i[:pnotE], 1e-10), 1e-10)
        else
            s / max(i[:pH] / max(i[:pnotE], 1e-10), 1e-10)
        end
    end),
]

println("\nPrior-DEPENDENT measures (use p(H) directly):")
for (name, formula) in prior_dependent
    r = evaluate_measure_detailed(formula, 99999)
    println("  $name: AUC = $(round(r.overall_auc, digits=4))")
end

println("\nPrior-INDEPENDENT measures (factor out p(H)):")
for (name, formula) in prior_independent
    r = evaluate_measure_detailed(formula, 99999)
    println("  $name: AUC = $(round(r.overall_auc, digits=4))")
end

println("""

The gap shows that for truth tracking, the prior carries predictive value.
This doesn't mean prior-independent measures are "wrong" — they serve a
different purpose (measuring evidential support independent of base rates).

IMPLICATIONS:
1. For PREDICTION tasks, consider prior-dependent formulas
2. For EVIDENTIAL REASONING, prior-independent measures remain appropriate
3. The best measure depends on your goal
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

println("="^70)
println("SUMMARY")
println("="^70)

winner = sorted_results[1]
println("""

BEST FORMULA FOR TRUTH TRACKING:
  $(winner[1])
  AUC = $(round(winner[2].overall_auc, digits=4)) [$(round(winner[2].ci_low, digits=3)), $(round(winner[2].ci_high, digits=3))]

KEY FINDINGS:
1. Prior-dependent formulas consistently outperform classical measures
2. The formula p(H|E) + p(H) - p(E|¬H) combines:
   - Posterior: p(H|E)
   - Prior: p(H) [adds predictive info]
   - Penalty: -p(E|¬H) [false positive rate]
   
3. The simpler formula p(H|E) / (p(¬H) + p(E|¬H)) offers good
   performance with better interpretability

THEORETICAL INSIGHT:
The success of prior-dependent formulas for truth tracking doesn't
undermine traditional confirmation measures — it reveals that
"tracking truth" and "measuring evidential support" are distinct tasks
that call for different measures.
""")

println("="^70)
println("Analysis complete!")
println("="^70)
