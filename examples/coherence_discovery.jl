#=
Coherence Measure Discovery via Symbolic Optimization
=====================================================

Using Angere's (2008) framework of truth-conduciveness to search for
coherence measures that optimally predict posterior probability of joint truth.

Key design choices:
  1. Multi-n: measures must generalise across set sizes n ∈ {2, 3, 4, 5}
  2. Stratified evaluation: Pearson r computed WITHIN each (n, reliability)
     stratum, then averaged. This gives GP a smooth fitness landscape:
     within a stratum, the posterior is a deterministic function of the
     state-description probabilities, so a perfect measure could reach r=1.
  3. Report τ_b at the end for comparison with Angere's literature.

Run:
    julia --project=. examples/coherence_discovery.jl

References:
  Angere, S. (2008). Coherence as a Heuristic. Mind, 117(465), 1–26.
  Bovens, L. & Hartmann, S. (2003). Bayesian Epistemology. OUP.
  Olsson, E. J. (2005). Against Coherence. OUP.
  Shogenji, T. (1999). Is Coherence Truth-conducive? Analysis, 59, 338–345.
=#

using SymbolicOptimization
using Random
using Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

"""Draw uniformly from the (k-1)-simplex via -log(Uniform)."""
function rand_simplex(rng::AbstractRNG, k::Int)
    x = [-log(rand(rng)) for _ in 1:k]
    return x ./ sum(x)
end

# ── Subset probability helpers (for computing Cd and O from state descriptions) ──

"""P(∧subset) given state description vector sp, where subset is a bitmask."""
function prob_conjunction(sp::Vector{Float64}, mask::Int, n_states::Int)
    s = 0.0
    @inbounds for k in 1:n_states
        if (k - 1) & mask == mask
            s += sp[k]
        end
    end
    return s
end

"""P(∨subset) given state description vector sp, where subset is a bitmask."""
function prob_disjunction(sp::Vector{Float64}, mask::Int, n_states::Int)
    s = 0.0
    @inbounds for k in 1:n_states
        if (k - 1) & mask != 0
            s += sp[k]
        end
    end
    return s
end

"""
Compute C_d (Douven & Meijs 2007): average difference measure over
all ordered pairs of non-empty non-overlapping subsets.

    C_d(S) = (1/[[S]]) Σ_{⟨S',S*⟩ ∈ [S]} [p(∧S' | ∧S*) - p(∧S')]
"""
function compute_cd_anyany(sp::Vector{Float64}, n::Int)
    n_states = 1 << n
    total = 0.0
    count = 0
    for s1 in 1:(n_states - 1)           # non-empty subsets as bitmasks
        p_s1 = prob_conjunction(sp, s1, n_states)
        for s2 in 1:(n_states - 1)
            s1 & s2 != 0 && continue      # must be non-overlapping
            p_s2 = prob_conjunction(sp, s2, n_states)
            if p_s2 > 1e-15
                p_both = prob_conjunction(sp, s1 | s2, n_states)
                total += p_both / p_s2 - p_s1
            end
            count += 1
        end
    end
    return count > 0 ? total / count : 0.0
end

"""
Compute Meijs (2006) overlap measure O(S): average of p(∧S')/p(∨S')
over all subsets S' of S with |S'| ≥ 2.
"""
function compute_overlap_meijs(sp::Vector{Float64}, n::Int)
    n_states = 1 << n
    total = 0.0
    count = 0
    for s in 1:(n_states - 1)
        count_ones(s) < 2 && continue     # need |S'| ≥ 2
        p_and = prob_conjunction(sp, s, n_states)
        p_or  = prob_disjunction(sp, s, n_states)
        if p_or > 1e-15
            total += p_and / p_or
        end
        count += 1
    end
    return count > 0 ? total / count : 0.0
end

"""
Compute mean pairwise conditional: (1/(n(n-1))) Σ_{i≠j} p(Aᵢ|Aⱼ).
"""
function compute_mean_pair_cond(p::Vector{Float64}, pair_joints::Vector{Float64}, n::Int)
    n < 2 && return 0.0
    total = 0.0
    count = 0
    idx = 0
    for i in 1:n
        for j in i+1:n
            idx += 1
            pij = pair_joints[idx]
            if p[j] > 1e-15
                total += pij / p[j]   # P(Aᵢ|Aⱼ)
                count += 1
            end
            if p[i] > 1e-15
                total += pij / p[i]   # P(Aⱼ|Aᵢ)
                count += 1
            end
        end
    end
    return count > 0 ? total / count : 0.0
end

"""Pearson correlation coefficient. Returns 0.0 for degenerate inputs."""
function pearson_r(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    n < 3 && return 0.0
    mx, my = mean(x), mean(y)
    num = 0.0; dx = 0.0; dy = 0.0
    @inbounds for i in 1:n
        xi = x[i] - mx
        yi = y[i] - my
        num += xi * yi
        dx += xi * xi
        dy += yi * yi
    end
    denom = sqrt(dx * dy)
    return denom > 1e-15 ? num / denom : 0.0
end

"""Kendall's τ-b rescaled to [0, 1]. Used for final evaluation only."""
function kendall_tau_b(scores::Vector{Float64}, posteriors::Vector{Float64})
    n = length(scores)
    concordant = 0; discordant = 0
    ties_s = 0; ties_p = 0
    @inbounds for i in 1:n-1
        si, vi = scores[i], posteriors[i]
        for j in i+1:n
            sj, vj = scores[j], posteriors[j]
            sd = si > sj ? 1 : (si < sj ? -1 : 0)
            vd = vi > vj ? 1 : (vi < vj ? -1 : 0)
            if sd == 0 && vd == 0
            elseif sd == 0; ties_s += 1
            elseif vd == 0; ties_p += 1
            elseif sd == vd; concordant += 1
            else;            discordant += 1
            end
        end
    end
    n0 = n * (n - 1) ÷ 2
    denom = sqrt(Float64(n0 - ties_s) * Float64(n0 - ties_p))
    τ = denom > 0.0 ? (concordant - discordant) / denom : 0.0
    return (τ + 1.0) / 2.0
end

# ═══════════════════════════════════════════════════════════════════════════════
# Testimonial System Generation (generalised to arbitrary n)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generate_system(rng, n; fixed_rel) → (ctx, posterior)

Generate a random testimonial system for `n` propositions at fixed reliability
`fixed_rel`, and compute P(&A|&E) via Angere's eq. 2.3.
"""
function generate_system(rng::AbstractRNG, n::Int; fixed_rel::Float64)
    n_states = 1 << n
    sp = rand_simplex(rng, n_states)
    rel = fill(fixed_rel, n)

    # Marginals P(Aᵢ)
    p = zeros(n)
    for i in 1:n
        mask = 1 << (i - 1)
        for j in 1:n_states
            if (j - 1) & mask != 0
                p[i] += sp[j]
            end
        end
    end

    # Pairwise joints P(Aᵢ ∧ Aⱼ)
    pair_joints = Float64[]
    for i in 1:n, j in i+1:n
        mask_ij = (1 << (i - 1)) | (1 << (j - 1))
        push!(pair_joints, sum(sp[k] for k in 1:n_states if (k - 1) & mask_ij == mask_ij))
    end

    # (n-1)-wise conjunctions P(A_{-i})
    full_mask = (1 << n) - 1
    conj_rest = Float64[]
    for i in 1:n
        rest_mask = full_mask & ~(1 << (i - 1))
        push!(conj_rest, sum(sp[k] for k in 1:n_states if (k - 1) & rest_mask == rest_mask))
    end

    # Aggregates
    pAnd  = sp[n_states]
    pOr   = 1.0 - sp[1]
    pNone = sp[1]
    prodP = prod(p)
    sumP  = sum(p)
    meanP = sumP / n

    n_pairs = length(pair_joints)
    sumPairJoints  = sum(pair_joints)
    prodPairJoints = prod(pair_joints)
    meanPairJoint  = n_pairs > 0 ? sumPairJoints / n_pairs : 0.0

    prodConjRest = prod(conj_rest)
    sumCondRest  = sum(cr > 1e-15 ? pAnd / cr : 0.0 for cr in conj_rest)
    meanCondRest = sumCondRest / n

    # New primitives: pairwise conditionals, Cd (any-any), Meijs overlap
    meanPairCond = compute_mean_pair_cond(p, pair_joints, n)
    cdAnyAny     = compute_cd_anyany(sp, n)
    overlapMeijs = compute_overlap_meijs(sp, n)

    # Credibilities & posterior (Angere eq. 2.2, 2.3)
    cred = [p[i] + fixed_rel * (1.0 - p[i]) for i in 1:n]

    numerator = 0.0; denominator = 0.0
    for j in 1:n_states
        prod_xi = 1.0; prod_cred_xi = 1.0
        for i in 1:n
            bit_set = (j - 1) & (1 << (i - 1)) != 0
            prod_xi      *= bit_set ? p[i] : (1.0 - p[i])
            prod_cred_xi *= bit_set ? cred[i] : (1.0 - cred[i])
        end
        cs_x = prod_xi > 1e-15 ? sp[j] / prod_xi : 0.0
        term = cs_x * prod_cred_xi
        denominator += term
        if j == n_states; numerator = term; end
    end
    posterior = denominator > 1e-15 ? numerator / denominator : 0.0

    ctx = Dict{Symbol,Any}(
        :pAnd => pAnd, :pOr => pOr, :pNone => pNone,
        :prodP => prodP, :sumP => sumP, :meanP => meanP,
        :sumPairJoints => sumPairJoints, :meanPairJoint => meanPairJoint,
        :prodPairJoints => prodPairJoints,
        :prodConjRest => prodConjRest, :meanCondRest => meanCondRest,
        :meanPairCond => meanPairCond,
        :cdAnyAny => cdAnyAny, :overlapMeijs => overlapMeijs,
        :nProps => Float64(n),
    )
    return (ctx=ctx, posterior=posterior)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset: stratified by (n, reliability)
# ═══════════════════════════════════════════════════════════════════════════════

struct Stratum
    n::Int
    reliability::Float64
    ctxs::Vector{Dict{Symbol,Any}}
    posteriors::Vector{Float64}
end

function generate_strata(rng::AbstractRNG;
        n_values=[2, 3, 4, 5],
        reliabilities=[0.3, 0.5, 0.7, 0.9],
        systems_per_stratum=300)
    strata = Stratum[]
    for n_val in n_values
        for rel in reliabilities
            systems = [generate_system(rng, n_val; fixed_rel=rel)
                       for _ in 1:systems_per_stratum]
            ctxs = [s.ctx for s in systems]
            posts = Float64[s.posterior for s in systems]
            push!(strata, Stratum(n_val, rel, ctxs, posts))
        end
    end
    return strata
end

# ═══════════════════════════════════════════════════════════════════════════════
# Known measures
# ═══════════════════════════════════════════════════════════════════════════════

prior_measure(ctx)  = ctx[:pAnd]
shogenji(ctx)       = ctx[:prodP] > 1e-15 ? ctx[:pAnd] / ctx[:prodP] : 0.0
olsson(ctx)         = ctx[:pOr]   > 1e-15 ? ctx[:pAnd] / ctx[:pOr]   : 0.0
log_shogenji(ctx)   = ctx[:prodP] > 1e-15 ? log(ctx[:pAnd] / ctx[:prodP]) : -20.0

function ewing_measure(ctx)
    d = ctx[:prodConjRest]
    n = Int(ctx[:nProps])
    return d > 1e-15 ? ctx[:pAnd]^n / d : 0.0
end

mean_cond(ctx) = ctx[:meanCondRest]
cd_douven_meijs(ctx) = ctx[:cdAnyAny]
overlap_meijs(ctx) = ctx[:overlapMeijs]

const KNOWN_MEASURES = [
    ("Prior",        prior_measure),
    ("Shogenji",     shogenji),
    ("Olsson",       olsson),
    ("log-Shogenji", log_shogenji),
    ("C_E",          ewing_measure),
    ("MeanCond",     mean_cond),
    ("C_d (D&M)",    cd_douven_meijs),
    ("O (Meijs)",    overlap_meijs),
]

"""Evaluate a measure's average Pearson r across strata."""
function eval_measure_pearson(mfn, strata::Vector{Stratum})
    r_values = Float64[]
    for s in strata
        scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
        if std(scores) > 1e-10
            push!(r_values, pearson_r(scores, s.posteriors))
        else
            push!(r_values, 0.0)
        end
    end
    return mean(r_values)
end

"""Evaluate a measure's average τ_b across strata."""
function eval_measure_tau(mfn, strata::Vector{Stratum})
    τ_values = Float64[]
    for s in strata
        scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
        if std(scores) > 1e-10
            push!(τ_values, kendall_tau_b(scores, s.posteriors))
        else
            push!(τ_values, 0.5)
        end
    end
    return mean(τ_values)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Detailed evaluation table
# ═══════════════════════════════════════════════════════════════════════════════

function print_detailed_table(strata, mfn_extra=nothing, extra_name="Discovered")
    n_values = sort(unique(s.n for s in strata))
    rels = sort(unique(s.reliability for s in strata))

    # Group strata by (n, rel)
    lookup = Dict((s.n, s.reliability) => s for s in strata)

    all_measures = copy(KNOWN_MEASURES)
    if mfn_extra !== nothing
        push!(all_measures, (extra_name, mfn_extra))
    end

    # Per-n table (averaged over reliabilities)
    println("\n  Per-n average Pearson r (averaged over reliabilities $(rels)):\n")
    print("  $(rpad("Measure", 14))")
    for nv in n_values; print(" │ n=$nv   "); end
    println(" │  avg")
    print("  $(repeat("─", 14))")
    for _ in n_values; print("─┼────────"); end
    println("─┼────────")

    for (name, mfn) in all_measures
        print("  $(rpad(name, 14))")
        per_n_avgs = Float64[]
        for nv in n_values
            r_vals = Float64[]
            for rel in rels
                s = lookup[(nv, rel)]
                scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
                push!(r_vals, std(scores) > 1e-10 ? pearson_r(scores, s.posteriors) : 0.0)
            end
            avg_r = mean(r_vals)
            push!(per_n_avgs, avg_r)
            print(" │ $(lpad(round(avg_r, digits=4), 6)) ")
        end
        println(" │ $(lpad(round(mean(per_n_avgs), digits=4), 6))")
    end

    # Per-reliability table (averaged over n)
    println("\n  Per-reliability average Pearson r (averaged over n = $(n_values)):\n")
    print("  $(rpad("Measure", 14))")
    for rel in rels; print(" │ r=$(rel) "); end
    println(" │  avg")
    print("  $(repeat("─", 14))")
    for _ in rels; print("─┼────────"); end
    println("─┼────────")

    for (name, mfn) in all_measures
        print("  $(rpad(name, 14))")
        per_rel_avgs = Float64[]
        for rel in rels
            r_vals = Float64[]
            for nv in n_values
                s = lookup[(nv, rel)]
                scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
                push!(r_vals, std(scores) > 1e-10 ? pearson_r(scores, s.posteriors) : 0.0)
            end
            avg_r = mean(r_vals)
            push!(per_rel_avgs, avg_r)
            print(" │ $(lpad(round(avg_r, digits=4), 6)) ")
        end
        println(" │ $(lpad(round(mean(per_rel_avgs), digits=4), 6))")
    end

    # Also report τ_b for comparison with Angere
    println("\n  Average Kendall τ_b (for comparison with Angere 2008):\n")
    for (name, mfn) in all_measures
        τ = eval_measure_tau(mfn, strata)
        println("  $(rpad(name, 14)) τ_b = $(round(τ, digits=4))")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Single optimization run with configurable seed group
# ═══════════════════════════════════════════════════════════════════════════════

"""
Seed groups force diversity across runs:
  :full            — all seeds including meanCondRest (baseline)
  :no_mc           — exclude meanCondRest and derivatives (forces alternative basins)
  :log_focus       — log-transformed measures only
  :prior_mods      — build from prior/Olsson direction
  :sigmoid         — x/(x+a) normalization motif from previous discoveries
  :discovered      — seed with previously discovered measures
  :mutual_support  — seeds from Douven & Meijs's Cd (pairwise/any-any support)
  :overlap         — seeds from Meijs's overlap measure
"""
function run_optimization(train_strata, run_seed;
                          population=100, generations=50,
                          seed_group::Symbol=:full)
    m = SymbolicModel()

    @variables(m,
        pAnd, pOr, pNone,
        prodP, sumP, meanP,
        sumPairJoints, meanPairJoint, prodPairJoints,
        prodConjRest, meanCondRest,
        meanPairCond, cdAnyAny, overlapMeijs,
        nProps,
    )

    @operators(m, binary=[+, -, *, /], unary=[log])
    @constants(m, -2.0..2.0, probability=0.15)

    @objective(m, Min, :neg_r, (tree, env) -> begin
        strata = env[:train_strata]
        r_values = Float64[]
        for s in strata
            scores = Float64[]
            sizehint!(scores, length(s.ctxs))
            for ctx in s.ctxs
                val = evaluate(tree, ctx)
                push!(scores, isfinite(val) ? val : 0.0)
            end
            if std(scores) < 1e-10
                push!(r_values, 0.0)
            else
                push!(r_values, pearson_r(scores, s.posteriors))
            end
        end
        return 1.0 - mean(r_values) + 0.0005 * count_nodes(tree)
    end)

    # ── Seed groups ──
    if seed_group == :full
        @seed(m, pAnd)
        @seed(m, pAnd / prodP)
        @seed(m, pAnd / pOr)
        @seed(m, log(pAnd / prodP))
        @seed(m, meanCondRest)
        @seed(m, log(meanCondRest))
        @seed(m, pAnd / prodConjRest)
        @seed(m, pAnd / meanPairJoint)
        @seed(m, prodP / pNone)
        @seed(m, meanPairJoint / meanP)
        @seed(m, meanCondRest * pAnd)

    elseif seed_group == :no_mc
        @seed(m, pAnd)
        @seed(m, pAnd / prodP)
        @seed(m, pAnd / pOr)
        @seed(m, log(pAnd / prodP))
        @seed(m, pAnd / prodConjRest)
        @seed(m, log(pAnd / prodConjRest))
        @seed(m, pAnd / meanPairJoint)
        @seed(m, prodP / pNone)
        @seed(m, meanPairJoint / meanP)
        @seed(m, meanPairJoint * pOr)
        @seed(m, pAnd / sumPairJoints)

    elseif seed_group == :log_focus
        @seed(m, log(pAnd / prodP))
        @seed(m, log(pAnd / pOr))
        @seed(m, log(meanCondRest))
        @seed(m, log(pAnd / prodConjRest))
        @seed(m, log(prodP / pNone))
        @seed(m, log(meanPairJoint / meanP))
        @seed(m, log(pAnd / meanPairJoint))

    elseif seed_group == :prior_mods
        @seed(m, pAnd)
        @seed(m, pAnd / pOr)
        @seed(m, pAnd + meanPairJoint)
        @seed(m, pAnd * meanPairJoint)
        @seed(m, pOr * meanPairJoint)
        @seed(m, pAnd / pNone)
        @seed(m, pOr - sumP + pAnd)
        @seed(m, sumPairJoints / sumP)
        @seed(m, prodPairJoints / prodP)

    elseif seed_group == :sigmoid
        # x/(x+a) normalization + log-shift motifs from discoveries
        @seed(m, pAnd / (pAnd + prodP))               # C_S/(C_S+1) core
        @seed(m, pAnd / (pAnd + prodP) + pAnd)        # Winner from round 1
        @seed(m, pAnd * meanP / (prodP + pAnd * meanP))  # Runner-up
        @seed(m, pAnd / (pAnd + pOr))                 # Sigmoid-Olsson variant
        @seed(m, meanCondRest / (meanCondRest + meanP))  # Sigmoid of conditional
        @seed(m, pAnd / (pAnd + prodConjRest))         # Sigmoid C_E variant
        @seed(m, log(pAnd / prodP + meanP))            # Log-shifted Shogenji (from winner)
        @seed(m, pAnd / (pAnd + pNone))                # Sigmoid vs all-false
        @seed(m, log(pAnd / (pAnd + prodP)))           # Log-sigmoid

    elseif seed_group == :discovered
        # Best measures from latest runs + building blocks
        @seed(m, pAnd / (pAnd + prodP) + pAnd)         # #2: C_S/(C_S+1) + P(&A)
        @seed(m, pAnd / (pAnd + prodP) + pAnd + meanCondRest) # #2 extended
        @seed(m, log(pAnd / prodP + meanP) * (sumP - prodConjRest))  # #1: new winner
        @seed(m, log(pAnd / prodP) + meanCondRest * sumP) # building block from #6
        @seed(m, cdAnyAny + meanCondRest + meanP)        # Cd + MC + meanP family
        @seed(m, cdAnyAny + meanCondRest + pAnd)         # Cd + MC + prior
        @seed(m, pAnd / (pAnd + prodP))                 # Core sigmoid
        @seed(m, sumP - prodConjRest)                     # Key building block
        @seed(m, log(pAnd / prodP + meanP))              # Log-shifted Shogenji

    elseif seed_group == :mutual_support
        # Primitives from Douven & Meijs (2007): Cd and pairwise support
        @seed(m, cdAnyAny)                               # C_d itself
        @seed(m, cdAnyAny + pAnd)                        # C_d + prior
        @seed(m, cdAnyAny + meanCondRest)                # C_d + one-all
        @seed(m, cdAnyAny + cdAnyAny + meanCondRest + meanP) # top pattern from run
        @seed(m, meanPairCond - meanP)                   # pairwise difference
        @seed(m, cdAnyAny * pOr + meanCondRest)          # Cd·P(∨A) + MC (#20)
        @seed(m, cdAnyAny + pAnd / (pAnd + prodP))      # C_d + sigmoid Shogenji
        @seed(m, cdAnyAny * pOr + meanPairCond)          # interaction term
        @seed(m, log(meanPairCond / meanP))              # log pairwise ratio

    elseif seed_group == :overlap
        # From Meijs (2006): overlap-based coherence
        @seed(m, overlapMeijs)                           # O(S) itself
        @seed(m, pAnd / pOr)                             # Olsson = 2-prop overlap
        @seed(m, overlapMeijs + cdAnyAny)                # Overlap + support
        @seed(m, overlapMeijs + meanCondRest)            # Overlap + one-all
        @seed(m, log(overlapMeijs))                      # log-overlap
        @seed(m, overlapMeijs + pAnd)                    # Overlap + prior
        @seed(m, overlapMeijs / (overlapMeijs + meanP))  # Sigmoid overlap
        @seed(m, pAnd / pOr + meanPairCond)              # Olsson + pairwise
    end

    @data(m, train_strata = train_strata)

    @config(m,
        population = population,
        generations = generations,
        max_depth = 8,
        max_nodes = 25,
        tournament_size = 5,
        parsimony_tolerance = 0.01,
        seed = run_seed,
    )

    optimize!(m)
    return m
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main — multiple independent restarts
# ═══════════════════════════════════════════════════════════════════════════════

function main(; seed=42,
                n_values=[2, 3, 4, 5],
                reliabilities=[0.3, 0.5, 0.7, 0.9],
                systems_per_stratum=300,
                n_restarts=48,
                pop_per_run=150,
                gen_per_run=100)

    rng = MersenneTwister(seed)

    n_strata = length(n_values) * length(reliabilities)
    n_total = n_strata * systems_per_stratum

    println("="^70)
    println("Coherence Measure Discovery (Angere's Framework)")
    println("="^70)
    println("\nParameters:")
    println("  Restarts: $n_restarts × (pop=$pop_per_run, gen=$gen_per_run)")
    println("  Set sizes n: $n_values")
    println("  Reliabilities: $reliabilities")
    println("  Systems per stratum: $systems_per_stratum")
    println("  Total strata: $n_strata  ($n_total systems)")
    println("  Fitness: average Pearson r(C, posterior) across strata")
    println()

    # ── Generate data ──
    println("Generating training data...")
    train_strata = generate_strata(rng; n_values, reliabilities, systems_per_stratum)

    test_rng = MersenneTwister(seed + 1000)
    println("Generating test data...")
    test_strata = generate_strata(test_rng; n_values, reliabilities, systems_per_stratum)

    # ── Baselines ──
    println("\nBaseline average Pearson r (across all $(n_strata) strata):")
    for (name, mfn) in KNOWN_MEASURES
        r_avg = eval_measure_pearson(mfn, test_strata)
        τ_avg = eval_measure_tau(mfn, test_strata)
        println("  $(rpad(name, 14)) r = $(round(r_avg, digits=4))   τ_b = $(round(τ_avg, digits=4))")
    end

    print_detailed_table(test_strata)

    # ══════════════════════════════════════════════════════════════════════════
    # Multiple independent restarts
    # ══════════════════════════════════════════════════════════════════════════

    println("\n" * "="^70)
    println("Running $n_restarts independent optimizations (stratified seed groups)...")
    println("="^70)

    # Assign seed groups: cycle through them so each gets multiple runs
    seed_groups = [:full, :no_mc, :log_focus, :prior_mods, :discovered,
                   :sigmoid, :mutual_support, :overlap]
    group_names = Dict(:full => "full", :no_mc => "no-MC", :log_focus => "log",
                       :prior_mods => "prior", :discovered => "disc", :sigmoid => "sigm",
                       :mutual_support => "C_d", :overlap => "O")

    all_solutions = Dict{String, Any}()

    for run in 1:n_restarts
        run_seed = seed + run * 137
        group = seed_groups[mod1(run, length(seed_groups))]
        gname = group_names[group]
        print("  Run $run/$n_restarts [$gname] (seed=$run_seed)... ")

        m = run_optimization(train_strata, run_seed;
                             population=pop_per_run, generations=gen_per_run,
                             seed_group=group)

        n_new = 0
        for ind in m.result.population
            expr = node_to_string(ind.tree)
            if !haskey(all_solutions, expr) || ind.objectives[1] < all_solutions[expr].obj
                all_solutions[expr] = (tree=ind.tree, obj=ind.objectives[1],
                                       run=run, group=gname)
                n_new += 1
            end
        end
        best_expr = best(m).expression
        best_obj = round(best(m).objectives[1], digits=4)
        println("best=$best_obj ($best_expr), $n_new new")
    end

    println("\n  Total unique solutions collected: $(length(all_solutions))")

    # ══════════════════════════════════════════════════════════════════════════
    # Evaluate all collected solutions on test data
    # ══════════════════════════════════════════════════════════════════════════

    println("\nEvaluating all solutions on test data...")
    ranked = []
    for (expr, sol) in all_solutions
        fn(ctx) = let v = evaluate(sol.tree, ctx); isfinite(v) ? v : 0.0 end
        r_avg = eval_measure_pearson(fn, test_strata)
        push!(ranked, (r=r_avg, expr=expr, tree=sol.tree, run=sol.run, group=sol.group))
    end
    sort!(ranked, by=r -> -r.r)

    # Best discovered
    best_sol = ranked[1]
    best_tree = best_sol.tree
    disc_fn(ctx) = let v = evaluate(best_tree, ctx); isfinite(v) ? v : 0.0 end
    disc_r = best_sol.r
    disc_τ = eval_measure_tau(disc_fn, test_strata)

    println("\n" * "="^70)
    println("RESULTS")
    println("="^70)
    println("\nDiscovered measure:")
    println("  $(best_sol.expr)")
    println("  (from run $(best_sol.run), group: $(best_sol.group))")
    println("\n  Average Pearson r: $(round(disc_r, digits=4))")
    println("  Average τ_b:      $(round(disc_τ, digits=4))")

    shog_r = eval_measure_pearson(shogenji, test_strata)
    ce_r   = eval_measure_pearson(ewing_measure, test_strata)
    mc_r   = eval_measure_pearson(mean_cond, test_strata)
    println("\n  Δr vs Shogenji:  $(round(disc_r - shog_r, digits=4))")
    println("  Δr vs C_E:       $(round(disc_r - ce_r, digits=4))")
    println("  Δr vs MeanCond:  $(round(disc_r - mc_r, digits=4))")

    print_detailed_table(test_strata, disc_fn, "Discovered")

    # ── Top solutions ──
    println("\n" * "="^70)
    println("Top 20 Solutions (by test Pearson r)")
    println("="^70)
    for i in 1:min(20, length(ranked))
        r = ranked[i]
        expr_short = length(r.expr) > 50 ? r.expr[1:47] * "..." : r.expr
        println("  $i. r=$(round(r.r, digits=4)) [$(r.group) #$(r.run)]: $expr_short")
    end

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: RIGOROUS STATISTICAL VALIDATION
    # ══════════════════════════════════════════════════════════════════════════

    println("\n\n" * "="^70)
    println("PHASE 2: STATISTICAL VALIDATION")
    println("="^70)

    # Validate top 5 unique measures + all known measures
    n_validate = min(5, length(ranked))
    top_trees = [(expr=ranked[i].expr, tree=ranked[i].tree) for i in 1:n_validate]

    # ── 2a: Extended n range (n = 2..8) ──
    println("\n── Generalisation to larger n (n = 2..8) ──")
    extended_n = [2, 3, 4, 5, 6, 7, 8]
    extended_rels = [0.3, 0.5, 0.7, 0.9]
    # Fewer systems per stratum for large n (2^n states gets expensive)
    ext_systems = [n <= 5 ? 300 : (n <= 7 ? 150 : 80) for n in extended_n]

    ext_rng = MersenneTwister(seed + 5000)
    println("  Generating extended test data...")
    ext_strata = Stratum[]
    for (ni, n_val) in enumerate(extended_n)
        for rel in extended_rels
            systems = [generate_system(ext_rng, n_val; fixed_rel=rel)
                       for _ in 1:ext_systems[ni]]
            ctxs = [s.ctx for s in systems]
            posts = Float64[s.posterior for s in systems]
            push!(ext_strata, Stratum(n_val, rel, ctxs, posts))
        end
    end

    # Group by n for per-n evaluation
    ext_by_n = Dict{Int, Vector{Stratum}}()
    for s in ext_strata
        push!(get!(ext_by_n, s.n, Stratum[]), s)
    end

    # Evaluate per n
    println("\n  Pearson r by n (averaged over reliabilities):\n")
    all_eval_measures = copy(KNOWN_MEASURES)
    for (i, t) in enumerate(top_trees)
        tree = t.tree
        fn = ctx -> let v = evaluate(tree, ctx); isfinite(v) ? v : 0.0 end
        push!(all_eval_measures, ("Top$i", fn))
    end

    print("  $(rpad("Measure", 14))")
    for nv in extended_n; print(" │  n=$nv "); end
    println(" │  avg")
    print("  $(repeat("─", 14))")
    for _ in extended_n; print("─┼───────"); end
    println("─┼───────")

    for (name, mfn) in all_eval_measures
        print("  $(rpad(name, 14))")
        r_per_n = Float64[]
        for nv in extended_n
            strata_n = ext_by_n[nv]
            r_vals = Float64[]
            for s in strata_n
                scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
                push!(r_vals, std(scores) > 1e-10 ? pearson_r(scores, s.posteriors) : 0.0)
            end
            avg = mean(r_vals)
            push!(r_per_n, avg)
            print(" │ $(lpad(round(avg, digits=3), 5)) ")
        end
        println(" │ $(lpad(round(mean(r_per_n), digits=3), 5))")
    end

    # Legend for Top measures
    println("\n  Legend:")
    for (i, t) in enumerate(top_trees)
        println("    Top$i = $(t.expr)")
    end

    # ── 2b: Bootstrap confidence intervals ──
    println("\n── Bootstrap Confidence Intervals (B=2000) ──\n")
    n_boot = 2000
    boot_rng = MersenneTwister(seed + 9000)

    # Use the original test strata for bootstrapping
    # For each bootstrap sample, resample systems within each stratum,
    # compute average r, collect distribution

    measures_to_boot = copy(KNOWN_MEASURES)
    for (i, t) in enumerate(top_trees)
        tree = t.tree
        fn = ctx -> let v = evaluate(tree, ctx); isfinite(v) ? v : 0.0 end
        push!(measures_to_boot, ("Top$i", fn))
    end

    boot_distributions = Dict{String, Vector{Float64}}()
    for (name, _) in measures_to_boot
        boot_distributions[name] = Float64[]
    end

    for b in 1:n_boot
        # Resample within each stratum
        boot_strata = Stratum[]
        for s in test_strata
            n_sys = length(s.posteriors)
            idx = [rand(boot_rng, 1:n_sys) for _ in 1:n_sys]
            push!(boot_strata, Stratum(s.n, s.reliability,
                                        s.ctxs[idx], s.posteriors[idx]))
        end

        for (name, mfn) in measures_to_boot
            r_avg = eval_measure_pearson(mfn, boot_strata)
            push!(boot_distributions[name], r_avg)
        end
    end

    # Print CIs
    println("  $(rpad("Measure", 14))   mean     [95% CI]            SE")
    println("  $(repeat("─", 60))")
    for (name, _) in measures_to_boot
        dist = sort(boot_distributions[name])
        m = mean(dist)
        lo = dist[max(1, round(Int, 0.025 * n_boot))]
        hi = dist[min(n_boot, round(Int, 0.975 * n_boot))]
        se = std(dist)
        println("  $(rpad(name, 14))   $(round(m, digits=4))   [$(round(lo, digits=4)), $(round(hi, digits=4))]   $(round(se, digits=4))")
    end

    # ── 2c: Pairwise bootstrap tests (Δr) ──
    println("\n── Pairwise Δr bootstrap tests (Top1 vs others) ──\n")
    top1_name = "Top1"
    top1_dist = boot_distributions[top1_name]
    println("  $(rpad("Comparison", 24))   ΔR̄      [95% CI]            p(Δr≤0)")
    println("  $(repeat("─", 70))")
    for (name, _) in measures_to_boot
        name == top1_name && continue
        other_dist = boot_distributions[name]
        deltas = top1_dist .- other_dist
        Δm = mean(deltas)
        sorted_d = sort(deltas)
        lo = sorted_d[max(1, round(Int, 0.025 * n_boot))]
        hi = sorted_d[min(n_boot, round(Int, 0.975 * n_boot))]
        p_neg = count(d -> d <= 0.0, deltas) / n_boot
        println("  $(rpad("Top1 - $name", 24))   $(round(Δm, digits=4))   [$(round(lo, digits=4)), $(round(hi, digits=4))]   $(round(p_neg, digits=4))")
    end

    # ── 2d: Kendall τ_b on extended n ──
    println("\n── Kendall τ_b by n (for comparison with Angere) ──\n")
    print("  $(rpad("Measure", 14))")
    for nv in extended_n; print(" │  n=$nv "); end
    println(" │  avg")
    print("  $(repeat("─", 14))")
    for _ in extended_n; print("─┼───────"); end
    println("─┼───────")

    for (name, mfn) in all_eval_measures
        print("  $(rpad(name, 14))")
        τ_per_n = Float64[]
        for nv in extended_n
            strata_n = ext_by_n[nv]
            τ_vals = Float64[]
            for s in strata_n
                scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
                push!(τ_vals, std(scores) > 1e-10 ? kendall_tau_b(scores, s.posteriors) : 0.5)
            end
            avg = mean(τ_vals)
            push!(τ_per_n, avg)
            print(" │ $(lpad(round(avg, digits=3), 5)) ")
        end
        println(" │ $(lpad(round(mean(τ_per_n), digits=3), 5))")
    end

    println("\n  Legend:")
    for (i, t) in enumerate(top_trees)
        println("    Top$i = $(t.expr)")
    end

    # ── 2e: Partial correlation controlling for P(&A) ──
    println("\n── Partial Pearson r controlling for P(&A) ──")
    println("  (Residual predictive power after removing prior's contribution)\n")

    """Partial correlation r_{XY·Z}: correlation of X and Y after linearly removing Z."""
    function partial_cor(x, y, z)
        n = length(x)
        n < 4 && return 0.0
        # Regress x on z → residuals
        z_mean = mean(z); x_mean = mean(x); y_mean = mean(y)
        sz2 = sum((zi - z_mean)^2 for zi in z)
        sz2 < 1e-15 && return pearson_r(x, y)  # z is constant → partial = raw
        # x residuals
        β_xz = sum((x[i] - x_mean) * (z[i] - z_mean) for i in 1:n) / sz2
        rx = [x[i] - x_mean - β_xz * (z[i] - z_mean) for i in 1:n]
        # y residuals
        β_yz = sum((y[i] - y_mean) * (z[i] - z_mean) for i in 1:n) / sz2
        ry = [y[i] - y_mean - β_yz * (z[i] - z_mean) for i in 1:n]
        # Correlation of residuals
        sr_x = sqrt(sum(v^2 for v in rx))
        sr_y = sqrt(sum(v^2 for v in ry))
        (sr_x < 1e-15 || sr_y < 1e-15) && return 0.0
        return sum(rx[i] * ry[i] for i in 1:n) / (sr_x * sr_y)
    end

    # Table: partial r by n (averaged over reliabilities)
    println("  Partial Pearson r by n (averaged over reliabilities):\n")
    print("  $(rpad("Measure", 14))")
    for nv in extended_n; print(" │  n=$nv "); end
    println(" │  avg")
    print("  $(repeat("─", 14))")
    for _ in extended_n; print("─┼───────"); end
    println("─┼───────")

    for (name, mfn) in all_eval_measures
        print("  $(rpad(name, 14))")
        pr_per_n = Float64[]
        for nv in extended_n
            strata_n = ext_by_n[nv]
            pr_vals = Float64[]
            for s in strata_n
                scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
                priors = Float64[ctx[:pAnd] for ctx in s.ctxs]
                pr = partial_cor(scores, s.posteriors, priors)
                push!(pr_vals, pr)
            end
            avg = mean(pr_vals)
            push!(pr_per_n, avg)
            print(" │ $(lpad(round(avg, digits=3), 5)) ")
        end
        println(" │ $(lpad(round(mean(pr_per_n), digits=3), 5))")
    end

    println("\n  Legend:")
    for (i, t) in enumerate(top_trees)
        println("    Top$i = $(t.expr)")
    end

    # Table: partial r by reliability (averaged over n)
    println("\n  Partial Pearson r by reliability (averaged over n = [2..8]):\n")
    ext_by_rel = Dict{Float64, Vector{Stratum}}()
    for s in ext_strata
        push!(get!(ext_by_rel, s.reliability, Stratum[]), s)
    end
    rels_sorted = sort(collect(keys(ext_by_rel)))

    print("  $(rpad("Measure", 14))")
    for r in rels_sorted; print(" │  r=$r "); end
    println(" │  avg")
    print("  $(repeat("─", 14))")
    for _ in rels_sorted; print("─┼────────"); end
    println("─┼───────")

    for (name, mfn) in all_eval_measures
        print("  $(rpad(name, 14))")
        pr_per_rel = Float64[]
        for rel in rels_sorted
            strata_r = ext_by_rel[rel]
            pr_vals = Float64[]
            for s in strata_r
                scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
                priors = Float64[ctx[:pAnd] for ctx in s.ctxs]
                pr = partial_cor(scores, s.posteriors, priors)
                push!(pr_vals, pr)
            end
            avg = mean(pr_vals)
            push!(pr_per_rel, avg)
            print(" │ $(lpad(round(avg, digits=3), 6)) ")
        end
        println(" │ $(lpad(round(mean(pr_per_rel), digits=3), 5))")
    end

    println("\n  Legend:")
    for (i, t) in enumerate(top_trees)
        println("    Top$i = $(t.expr)")
    end

    # Decomposition: R² from prior alone vs R² from measure alone vs R² from both
    println("\n── Variance decomposition: Prior vs Coherence ──")
    println("  (R² of posterior ~ P(&A), posterior ~ C, and posterior ~ P(&A) + C)\n")
    println("  $(rpad("Measure", 14)) │ R²(prior) │ R²(C)   │ R²(both) │ ΔR²(C|prior)")
    println("  $(repeat("─", 14))─┼───────────┼─────────┼──────────┼─────────────")

    for (name, mfn) in all_eval_measures
        r2_prior_all = Float64[]
        r2_C_all = Float64[]
        r2_both_all = Float64[]
        for s in ext_strata
            scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
            priors = Float64[ctx[:pAnd] for ctx in s.ctxs]
            posts = s.posteriors

            r_py = std(priors) > 1e-10 ? pearson_r(priors, posts) : 0.0
            r_cy = std(scores) > 1e-10 ? pearson_r(scores, posts) : 0.0
            # For R²(both), use partial correlation to get R² = 1 - (1-r²_py)(1-r²_cy·p)
            pr_cy = partial_cor(scores, posts, priors)

            push!(r2_prior_all, r_py^2)
            push!(r2_C_all, r_cy^2)
            push!(r2_both_all, 1.0 - (1.0 - r_py^2) * (1.0 - pr_cy^2))
        end
        r2p = mean(r2_prior_all)
        r2c = mean(r2_C_all)
        r2b = mean(r2_both_all)
        delta = r2b - r2p  # Incremental R² of coherence beyond prior
        println("  $(rpad(name, 14)) │ $(lpad(round(r2p, digits=4), 8))  │ $(lpad(round(r2c, digits=4), 6))  │ $(lpad(round(r2b, digits=4), 7))  │ $(lpad(round(delta, digits=4), 10))")
    end

    println("\n  Legend:")
    for (i, t) in enumerate(top_trees)
        println("    Top$i = $(t.expr)")
    end

    return ranked
end

ranked = main(seed=42)
