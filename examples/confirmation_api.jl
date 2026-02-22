#=
Confirmation Measure Discovery — New API Version
=================================================

This example shows how to discover confirmation measures using the new
JuMP-inspired API. Compare with confirmation_constrained_search.jl for
the low-level version.

The goal: Find formulas c(H,E) that measure how much evidence E confirms
or disconfirms hypothesis H, optimizing for truth-tracking (AUC) while
satisfying theoretical constraints (neutrality, directionality, symmetry).

Run:
    julia --project=. examples/confirmation_api.jl
=#

using SymbolicOptimization
using Random
using Statistics
using Distributions

# ═══════════════════════════════════════════════════════════════════════════════
# Step function for piecewise constructions
# ═══════════════════════════════════════════════════════════════════════════════
# step(x) = 1 if x ≥ 0, else 0
# Enables: step(cond) * A + (1 - step(cond)) * B  =  "if cond ≥ 0 then A else B"
step_func(x::Real) = x >= 0.0 ? 1.0 : 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation (matching the full TC simulation environment)
# ═══════════════════════════════════════════════════════════════════════════════

function generate_trial(rng::AbstractRNG, n_worlds::Int)
    @assert n_worlds > 3
    # Random hypothesis and evidence as subsets of worlds
    n_H = rand(rng, 1:n_worlds-1)
    n_E = rand(rng, 1:n_worlds-1)

    H_worlds = Set(randperm(rng, n_worlds)[1:n_H])
    E_worlds = Set(randperm(rng, n_worlds)[1:n_E])

    # Actual world drawn uniformly from E (veridical evidence)
    actual = rand(rng, collect(E_worlds))

    # Probability mass function from flat Dirichlet
    π = rand(rng, Dirichlet(n_worlds, 1.0))

    all_w = Set(1:n_worlds)
    notH = setdiff(all_w, H_worlds)
    notE = setdiff(all_w, E_worlds)

    # Marginals
    pH      = sum(π[w] for w in H_worlds)
    pE      = sum(π[w] for w in E_worlds)
    pnotH   = sum(π[w] for w in notH)
    pnotE   = sum(π[w] for w in notE)

    # Joints
    pH_and_E     = sum(π[w] for w in H_worlds ∩ E_worlds; init=0.0)
    pnotH_and_E  = sum(π[w] for w in notH ∩ E_worlds; init=0.0)
    pH_and_notE  = sum(π[w] for w in H_worlds ∩ notE; init=0.0)
    pnotH_and_notE = sum(π[w] for w in notH ∩ notE; init=0.0)

    # Conditionals (with safe fallbacks)
    pH_E     = pE > 1e-10 ? pH_and_E / pE : 0.5
    pnotH_E  = pE > 1e-10 ? pnotH_and_E / pE : 0.5
    pH_notE  = pnotE > 1e-10 ? pH_and_notE / pnotE : 0.5
    pnotH_notE = pnotE > 1e-10 ? pnotH_and_notE / pnotE : 0.5
    pE_H     = pH > 1e-10 ? pH_and_E / pH : 0.5
    pnotE_H  = pH > 1e-10 ? pH_and_notE / pH : 0.5
    pE_notH  = pnotH > 1e-10 ? pnotH_and_E / pnotH : 0.5
    pnotE_notH = pnotH > 1e-10 ? pnotH_and_notE / pnotH : 0.5

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

function generate_neutral_trial(rng::AbstractRNG)
    pH = rand(rng) * 0.8 + 0.1
    pE = rand(rng) * 0.8 + 0.1
    pnotH = 1 - pH
    pnotE = 1 - pE
    # Independence: P(H|E) = P(H), P(E|H) = P(E)
    pH_E = pH;       pnotH_E = pnotH
    pH_notE = pH;    pnotH_notE = pnotH
    pE_H = pE;       pnotE_H = pnotE
    pE_notH = pE;    pnotE_notH = pnotE
    pH_and_E = pH * pE
    pH_and_notE = pH * pnotE
    pnotH_and_E = pnotH * pE
    pnotH_and_notE = pnotH * pnotE

    return Dict{Symbol,Any}(
        :pH => pH, :pE => pE, :pnotH => pnotH, :pnotE => pnotE,
        :pH_and_E => pH_and_E, :pH_and_notE => pH_and_notE,
        :pnotH_and_E => pnotH_and_E, :pnotH_and_notE => pnotH_and_notE,
        :pH_E => pH_E, :pnotH_E => pnotH_E,
        :pH_notE => pH_notE, :pnotH_notE => pnotH_notE,
        :pE_H => pE_H, :pnotE_H => pnotE_H,
        :pE_notH => pE_notH, :pnotE_notH => pnotE_notH,
    )
end

function generate_entails_H_trial(rng::AbstractRNG)
    # E ⊨ H: every E-world is an H-world → P(H|E) = 1
    pH = rand(rng) * 0.8 + 0.1
    pE = rand(rng) * pH  # E ⊆ H → P(E) ≤ P(H)
    pnotH = 1 - pH; pnotE = 1 - pE
    pH_and_E = pE; pnotH_and_E = 0.0
    pH_and_notE = pH - pE; pnotH_and_notE = pnotH

    pH_E = 1.0; pnotH_E = 0.0
    pH_notE = pnotE > 1e-10 ? pH_and_notE / pnotE : 0.5
    pnotH_notE = pnotE > 1e-10 ? pnotH_and_notE / pnotE : 0.5
    pE_H = pH > 1e-10 ? pE / pH : 0.5
    pnotE_H = pH > 1e-10 ? (pH - pE) / pH : 0.5
    pE_notH = 0.0; pnotE_notH = 1.0

    return Dict{Symbol,Any}(
        :pH => pH, :pE => pE, :pnotH => pnotH, :pnotE => pnotE,
        :pH_and_E => pH_and_E, :pH_and_notE => pH_and_notE,
        :pnotH_and_E => pnotH_and_E, :pnotH_and_notE => pnotH_and_notE,
        :pH_E => pH_E, :pnotH_E => pnotH_E,
        :pH_notE => pH_notE, :pnotH_notE => pnotH_notE,
        :pE_H => pE_H, :pnotE_H => pnotE_H,
        :pE_notH => pE_notH, :pnotE_notH => pnotE_notH,
    )
end

function generate_entails_notH_trial(rng::AbstractRNG)
    # E ⊨ ¬H: every E-world is a ¬H-world → P(H|E) = 0
    pH = rand(rng) * 0.8 + 0.1
    pnotH = 1 - pH
    pE = rand(rng) * pnotH  # E ⊆ ¬H → P(E) ≤ P(¬H)
    pnotE = 1 - pE
    pH_and_E = 0.0; pnotH_and_E = pE
    pH_and_notE = pH; pnotH_and_notE = pnotH - pE

    pH_E = 0.0; pnotH_E = 1.0
    pH_notE = pnotE > 1e-10 ? pH / pnotE : 0.5
    pnotH_notE = pnotE > 1e-10 ? (pnotH - pE) / pnotE : 0.5
    pE_H = 0.0; pnotE_H = 1.0
    pE_notH = pnotH > 1e-10 ? pE / pnotH : 0.5
    pnotE_notH = pnotH > 1e-10 ? (pnotH - pE) / pnotH : 0.5

    return Dict{Symbol,Any}(
        :pH => pH, :pE => pE, :pnotH => pnotH, :pnotE => pnotE,
        :pH_and_E => pH_and_E, :pH_and_notE => pH_and_notE,
        :pnotH_and_E => pnotH_and_E, :pnotH_and_notE => pnotH_and_notE,
        :pH_E => pH_E, :pnotH_E => pnotH_E,
        :pH_notE => pH_notE, :pnotH_notE => pnotH_notE,
        :pE_H => pE_H, :pnotE_H => pnotE_H,
        :pE_notH => pE_notH, :pnotE_notH => pnotE_notH,
    )
end

function generate_confirming_trial(rng::AbstractRNG, n_worlds::Int)
    for _ in 1:100
        trial = generate_trial(rng, n_worlds)
        trial.pH_E > trial.pH + 0.01 && return trial
    end
    return generate_trial(rng, n_worlds)
end

function generate_disconfirming_trial(rng::AbstractRNG, n_worlds::Int)
    for _ in 1:100
        trial = generate_trial(rng, n_worlds)
        trial.pH_E < trial.pH - 0.01 && return trial
    end
    return generate_trial(rng, n_worlds)
end

function trial_to_dict(trial)
    Dict{Symbol,Any}(
        :pH => trial.pH, :pE => trial.pE, :pnotH => trial.pnotH, :pnotE => trial.pnotE,
        :pH_and_E => trial.pH_and_E, :pH_and_notE => trial.pH_and_notE,
        :pnotH_and_E => trial.pnotH_and_E, :pnotH_and_notE => trial.pnotH_and_notE,
        :pH_E => trial.pH_E, :pnotH_E => trial.pnotH_E,
        :pH_notE => trial.pH_notE, :pnotH_notE => trial.pnotH_notE,
        :pE_H => trial.pE_H, :pnotE_H => trial.pnotE_H,
        :pE_notH => trial.pE_notH, :pnotE_notH => trial.pnotE_notH,
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Constraint checking helpers
# ═══════════════════════════════════════════════════════════════════════════════

function compute_auc(scores::Vector{Float64}, labels)
    pos_scores = scores[labels .== 1.0]
    neg_scores = scores[labels .== 0.0]
    n_pos = length(pos_scores); n_neg = length(neg_scores)
    (n_pos == 0 || n_neg == 0) && return 0.5
    u = 0.0
    for ps in pos_scores, ns in neg_scores
        u += ps > ns ? 1.0 : ps == ns ? 0.5 : 0.0
    end
    return u / (n_pos * n_neg)
end

function check_neutrality(tree, evaluate_fn, neutral_X; tolerance=0.05)
    n = 0
    for x in neutral_X
        val = evaluate_fn(tree, x)
        isfinite(val) && abs(val) < tolerance && (n += 1)
    end
    n / length(neutral_X)
end

function check_directionality(tree, evaluate_fn, confirming_X, disconfirming_X)
    correct = 0; total = 0
    for x in confirming_X
        val = evaluate_fn(tree, x)
        isfinite(val) || continue
        total += 1; val > 0 && (correct += 1)
    end
    for x in disconfirming_X
        val = evaluate_fn(tree, x)
        isfinite(val) || continue
        total += 1; val < 0 && (correct += 1)
    end
    total > 0 ? correct / total : 0.0
end

function check_symmetric_logicality(tree, evaluate_fn, entails_H_X, entails_notH_X)
    vals_H = Float64[]; vals_notH = Float64[]
    for x in entails_H_X
        val = evaluate_fn(tree, x)
        isfinite(val) && push!(vals_H, val)
    end
    for x in entails_notH_X
        val = evaluate_fn(tree, x)
        isfinite(val) && push!(vals_notH, val)
    end
    (isempty(vals_H) || isempty(vals_notH)) && return 0.0
    frac_H_pos = mean(vals_H .> 0)
    frac_notH_neg = mean(vals_notH .< 0)
    mean_abs_H = mean(abs.(vals_H))
    mean_abs_notH = mean(abs.(vals_notH))
    symmetry = (mean_abs_H + mean_abs_notH > 1e-10) ?
        1 - abs(mean_abs_H - mean_abs_notH) / (mean_abs_H + mean_abs_notH) : 0.0
    0.5 * (frac_H_pos + frac_notH_neg) * symmetry
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main(; seed=42, n_main=3000, population=500, generations=150,
                min_neutrality=0.95, min_directionality=0.95, min_symmetry=0.90)
    rng = MersenneTwister(seed)
    world_range = 5:100

    println("="^70)
    println("Confirmation Measure Discovery (JuMP-style API)")
    println("="^70)
    println("\nParameters:")
    println("  Population: $population")
    println("  Generations: $generations")
    println("  Main trials: $n_main")
    println("  Min neutrality: $min_neutrality")
    println("  Min directionality: $min_directionality")
    println("  Min symmetry: $min_symmetry")
    println()

    # ── Generate datasets ──
    println("Generating data...")

    main_trials = [generate_trial(rng, rand(rng, world_range)) for _ in 1:n_main]
    main_X = [trial_to_dict(t) for t in main_trials]
    main_y = Float64[t.H_true ? 1.0 : 0.0 for t in main_trials]

    neutral_X    = [generate_neutral_trial(rng) for _ in 1:200]
    entails_H_X  = [generate_entails_H_trial(rng) for _ in 1:200]
    entails_notH_X = [generate_entails_notH_trial(rng) for _ in 1:200]
    confirming_X = [trial_to_dict(generate_confirming_trial(rng, rand(rng, world_range))) for _ in 1:200]
    disconfirming_X = [trial_to_dict(generate_disconfirming_trial(rng, rand(rng, world_range))) for _ in 1:200]

    # Separate test set
    test_rng = MersenneTwister(seed + 1000)
    test_trials = [generate_trial(test_rng, rand(test_rng, world_range)) for _ in 1:n_main]
    test_X = [trial_to_dict(t) for t in test_trials]
    test_y = Float64[t.H_true ? 1.0 : 0.0 for t in test_trials]
    test_neutral = [generate_neutral_trial(test_rng) for _ in 1:200]
    test_entails_H = [generate_entails_H_trial(test_rng) for _ in 1:200]
    test_entails_notH = [generate_entails_notH_trial(test_rng) for _ in 1:200]
    test_confirming = [trial_to_dict(generate_confirming_trial(test_rng, rand(test_rng, world_range))) for _ in 1:200]
    test_disconfirming = [trial_to_dict(generate_disconfirming_trial(test_rng, rand(test_rng, world_range))) for _ in 1:200]

    println("  Training: $(length(main_X)) main trials (worlds ∈ $world_range)")
    println("  Testing:  $(length(test_X)) main trials")
    println("  + 200 each of neutral, entails-H, entails-¬H, confirming, disconfirming")
    println()

    # ── Define the model ──
    m = SymbolicModel()

    # Full variable set — all 16 probabilities
    @variables(m,
        pH, pE, pnotH, pnotE,
        pH_and_E, pH_and_notE, pnotH_and_E, pnotH_and_notE,
        pH_E, pnotH_E, pH_notE, pnotH_notE,
        pE_H, pnotE_H, pE_notH, pnotE_notH
    )

    # Grammar: +, -, *, / plus step_func (piecewise) and log (logarithmic)
    @operators(m, binary=[+, -, *, /], unary=[step_func, log])

    # No random constants — measures must be purely structural
    @constants(m, nothing)

    # ── Objective 1: AUC + constraint penalties ──
    @objective(m, Min, :auc_penalty, (tree, env) -> begin
        main_X = env[:main_X];  main_y = env[:main_y]
        neut   = env[:neutral_X]
        conf_X = env[:confirming_X]; disconf_X = env[:disconfirming_X]
        ent_H  = env[:entails_H_X]; ent_notH = env[:entails_notH_X]
        min_neut = env[:min_neutrality]
        min_dir  = env[:min_directionality]
        min_sym  = env[:min_symmetry]

        # Evaluate on main trials
        scores = Float64[]
        for x in main_X
            val = evaluate(tree, x)
            push!(scores, isfinite(val) ? val : 0.0)
        end

        # Skip degenerate
        std(scores) < 1e-10 && return 1.0

        auc = compute_auc(scores, main_y)

        eval_fn = (t, x) -> evaluate(t, x)
        neutrality     = check_neutrality(tree, eval_fn, neut)
        directionality = check_directionality(tree, eval_fn, conf_X, disconf_X)
        symmetry       = check_symmetric_logicality(tree, eval_fn, ent_H, ent_notH)

        # Harsh soft penalties
        penalty = 0.0
        neutrality     < min_neut && (penalty += 1.0 * (min_neut - neutrality))
        directionality < min_dir  && (penalty += 1.5 * (min_dir - directionality))
        symmetry       < min_sym  && (penalty += 0.8 * (min_sym - symmetry))

        # Small complexity bonus
        complexity_bonus = 0.002 * count_nodes(tree)

        return 1.0 - auc + penalty + complexity_bonus
    end)

    # ── Objective 2: Complexity ──
    @objective(m, Min, :complexity)

    # ── Seed with known confirmation measures ──
    # Simple measures
    @seed(m, pH_E - pH)                              # d: difference
    @seed(m, pH_E - pH_notE)                         # s: Christensen
    @seed(m, log(pE_H / pE_notH))                   # l: Good's log-likelihood
    @seed(m, (pE_H - pE_notH) / (pE_H + pE_notH))  # k: Kemeny-Oppenheim
    @seed(m, (pH_E - pH) / pnotH)                    # g: Rips
    @seed(m, pE_H / pE_notH)                         # Likelihood ratio (raw)
    @seed(m, log(pH_E / pnotH_E))                    # Log posterior odds

    # z: Crupi et al. (piecewise via step_func)
    # z = step(d)·(d/p(¬H)) + (1 - step(d))·(d/p(H))  where d = p(H|E) - p(H)
    # Note: pH/pH encodes 1.0 without using constants (constants are disabled)
    @seed(m, step_func(pH_E - pH) * ((pH_E - pH) / pnotH) +
            (pH / pH - step_func(pH_E - pH)) * ((pH_E - pH) / pH))

    # ch: Cheng (piecewise via step_func)
    # ch = step(s)·(s/p(¬H|¬E)) + (1 - step(s))·(s/p(H|¬E))  where s = p(H|E) - p(H|¬E)
    @seed(m, step_func(pH_E - pH_notE) * ((pH_E - pH_notE) / pnotH_notE) +
            (pH / pH - step_func(pH_E - pH_notE)) * ((pH_E - pH_notE) / pH_notE))

    # Piecewise variant of d (normalised by priors)
    @seed(m, step_func(pH_E - pH) * ((pH_E - pH) / pnotH) +
            (pH / pH - step_func(pH_E - pH)) * ((pH_E - pH) / pH))

    # Piecewise variant of s (normalised by priors)
    @seed(m, step_func(pH_E - pH_notE) * ((pH_E - pH_notE) / pnotH) +
            (pH / pH - step_func(pH_E - pH_notE)) * ((pH_E - pH_notE) / pH))

    # ── Data ──
    @data(m,
        main_X = main_X, main_y = main_y,
        neutral_X = neutral_X,
        confirming_X = confirming_X, disconfirming_X = disconfirming_X,
        entails_H_X = entails_H_X, entails_notH_X = entails_notH_X,
        min_neutrality = min_neutrality,
        min_directionality = min_directionality,
        min_symmetry = min_symmetry,
    )

    # ── Configuration ──
    @config(m,
        population = population,
        generations = generations,
        max_depth = 8,       # Needs depth for step(cond)*A + (1-step(cond))*B
        max_nodes = 40,
        tournament_size = 7,
        parsimony_tolerance = 0.01,
        seed = seed,
    )

    println("Model:")
    println(m)

    # ── Optimize! ──
    println("\nRunning NSGA-II optimization...")
    optimize!(m)

    # ── Results ──
    best_tree = best(m).tree

    # Evaluate on TEST set
    eval_fn = (t, x) -> evaluate(t, x)
    test_scores = Float64[]
    for x in test_X
        val = evaluate(best_tree, x)
        push!(test_scores, isfinite(val) ? val : 0.0)
    end
    test_auc = compute_auc(test_scores, test_y)
    test_neut = check_neutrality(best_tree, eval_fn, test_neutral)
    test_dir  = check_directionality(best_tree, eval_fn, test_confirming, test_disconfirming)
    test_sym  = check_symmetric_logicality(best_tree, eval_fn, test_entails_H, test_entails_notH)

    println("\n" * "="^70)
    println("Best Solution Analysis")
    println("="^70)
    println("\nFormula: $(best(m).expression)")
    println("\nTest Set Results:")
    println("  AUC:            $(round(test_auc, digits=4))")
    println("  Neutrality:     $(round(test_neut, digits=4)) (min: $min_neutrality)")
    println("  Directionality: $(round(test_dir, digits=4)) (min: $min_directionality)")
    println("  Symmetry:       $(round(test_sym, digits=4)) (min: $min_symmetry)")

    # Compare with z and ch on the test set
    println("\nComparison with existing measures:\n")
    println("  Measure     │   AUC   │ Neutrality │ Directionality │ Symmetry")
    println("  ────────────┼─────────┼────────────┼────────────────┼──────────")

    for (name, measure_fn) in [
        ("z", ctx -> begin
            d = ctx[:pH_E] - ctx[:pH]
            d >= 0 ? d / ctx[:pnotH] : d / ctx[:pH]
        end),
        ("ch", ctx -> begin
            s = ctx[:pH_E] - ctx[:pH_notE]
            s >= 0 ? s / ctx[:pnotH_notE] : s / ctx[:pH_notE]
        end),
    ]
        scores = Float64[measure_fn(x) for x in test_X]
        replace!(x -> isfinite(x) ? x : 0.0, scores)
        auc_val = compute_auc(scores, test_y)

        # Wrap as a simple evaluator for constraint checks
        measure_tree_fn = (_, x) -> measure_fn(x)
        neut_val = check_neutrality(nothing, measure_tree_fn, test_neutral)
        dir_val  = check_directionality(nothing, measure_tree_fn, test_confirming, test_disconfirming)
        sym_val  = check_symmetric_logicality(nothing, measure_tree_fn, test_entails_H, test_entails_notH)

        println("  $(rpad(name, 12))│ $(lpad(round(auc_val, digits=4), 7)) │   $(lpad(round(neut_val, digits=4), 6))   │     $(lpad(round(dir_val, digits=4), 6))     │  $(lpad(round(sym_val, digits=4), 6))")
    end

    b = best(m)
    println("  $(rpad("Discovered", 12))│ $(lpad(round(test_auc, digits=4), 7)) │   $(lpad(round(test_neut, digits=4), 6))   │     $(lpad(round(test_dir, digits=4), 6))     │  $(lpad(round(test_sym, digits=4), 6))")

    z_scores = Float64[let d = x[:pH_E]-x[:pH]; d >= 0 ? d/x[:pnotH] : d/x[:pH] end for x in test_X]
    ch_scores = Float64[let s = x[:pH_E]-x[:pH_notE]; s >= 0 ? s/x[:pnotH_notE] : s/x[:pH_notE] end for x in test_X]
    println("\n  Δ AUC vs z:  $(round(test_auc - compute_auc(z_scores, test_y), digits=4))")
    println("  Δ AUC vs ch: $(round(test_auc - compute_auc(ch_scores, test_y), digits=4))")

    println("\n" * "="^70)
    println("FINAL RESULT")
    println("="^70)
    println("\nDiscovered formula:")
    println("  $(b.expression)")

    println("\nPareto Front:")
    front = pareto_front(m)
    seen = Set{String}()
    for (i, sol) in enumerate(sort(front, by=s -> s.objectives[1]))
        sol.expression ∈ seen && continue
        push!(seen, sol.expression)
        println("  $i. AUC=$(round(1.0 - sol.objectives[1], digits=4)), nodes=$(Int(sol.objectives[2])): $(sol.expression[1:min(end,75)])...")
        length(seen) >= 10 && break
    end

    return m
end

# Run
m = main(seed=42, n_main=3000, population=500, generations=150)
