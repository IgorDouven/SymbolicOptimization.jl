#=
Coherence Measure Evaluation & Comparison
==========================================

Standalone evaluation of discovered and known coherence measures.
No symbolic optimization — just generates test data and runs comprehensive
statistical comparisons, including partial correlation analysis controlling
for P(&A) to disentangle coherence from prior probability.

Run:
    julia --project=. examples/coherence_evaluation.jl

References:
  Angere, S. (2008). Coherence as a Heuristic. Mind, 117(465), 1–26.
  Douven, I. & Meijs, W. (2007). Measuring Coherence. Synthese, 156, 405–425.
  Meijs, W. (2006). Coherence as Generalized Logical Equivalence. Erkenntnis, 64, 231–252.
  Shogenji, T. (1999). Is Coherence Truth-conducive? Analysis, 59, 338–345.
=#

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
"""
function compute_cd_anyany(sp::Vector{Float64}, n::Int)
    n_states = 1 << n
    total = 0.0
    count = 0
    for s1 in 1:(n_states - 1)
        p_s1 = prob_conjunction(sp, s1, n_states)
        for s2 in 1:(n_states - 1)
            s1 & s2 != 0 && continue
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
        count_ones(s) < 2 && continue
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
                total += pij / p[j]
                count += 1
            end
            if p[i] > 1e-15
                total += pij / p[i]
                count += 1
            end
        end
    end
    return count > 0 ? total / count : 0.0
end

"""Pearson correlation coefficient."""
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

"""Kendall's τ-b rescaled to [0, 1]."""
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

"""Partial correlation r_{XY·Z}: correlation of X and Y after linearly removing Z."""
function partial_cor(x, y, z)
    n = length(x)
    n < 4 && return 0.0
    z_mean = mean(z); x_mean = mean(x); y_mean = mean(y)
    sz2 = sum((zi - z_mean)^2 for zi in z)
    sz2 < 1e-15 && return pearson_r(x, y)
    β_xz = sum((x[i] - x_mean) * (z[i] - z_mean) for i in 1:n) / sz2
    rx = [x[i] - x_mean - β_xz * (z[i] - z_mean) for i in 1:n]
    β_yz = sum((y[i] - y_mean) * (z[i] - z_mean) for i in 1:n) / sz2
    ry = [y[i] - y_mean - β_yz * (z[i] - z_mean) for i in 1:n]
    sr_x = sqrt(sum(v^2 for v in rx))
    sr_y = sqrt(sum(v^2 for v in ry))
    (sr_x < 1e-15 || sr_y < 1e-15) && return 0.0
    return sum(rx[i] * ry[i] for i in 1:n) / (sr_x * sr_y)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Testimonial System Generation
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

struct Stratum
    n::Int
    reliability::Float64
    ctxs::Vector{Dict{Symbol,Any}}
    posteriors::Vector{Float64}
end

# ═══════════════════════════════════════════════════════════════════════════════
# Known measures from the literature
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

# ═══════════════════════════════════════════════════════════════════════════════
# Discovered measures (from symbolic optimization runs)
# ═══════════════════════════════════════════════════════════════════════════════

# Top1: log(C_S + meanP) × (ΣP - ΠP(Aᵢ|A₋ᵢ))
# Pearson champion, near-perfect generalization across n
function discovered_top1(ctx)
    cs = ctx[:prodP] > 1e-15 ? ctx[:pAnd] / ctx[:prodP] : 0.0
    left = log(cs + ctx[:meanP])
    right = ctx[:sumP] - ctx[:prodConjRest]
    v = left * right
    return isfinite(v) ? v : 0.0
end

# Top2: C_S/(0.616·(C_S+1)) + meanCondRest + P(&A)
# τ-b champion, rescaled sigmoid
function discovered_top2(ctx)
    cs = ctx[:prodP] > 1e-15 ? ctx[:pAnd] / ctx[:prodP] : 0.0
    sigmoid = ctx[:pAnd] / ((ctx[:pAnd] + ctx[:prodP]) * 0.616)
    v = sigmoid + ctx[:meanCondRest] + ctx[:pAnd]
    return isfinite(v) ? v : 0.0
end

# Top4: C_S/(C_S+1) + P(&A) + meanCondRest/n
# Most interpretable: equal-weight combination of relative overlap,
# prior probability, and mutual support (all on [0,1] scale)
function discovered_top4(ctx)
    cs = ctx[:prodP] > 1e-15 ? ctx[:pAnd] / ctx[:prodP] : 0.0
    sigmoid = cs / (cs + 1.0)
    v = sigmoid + ctx[:pAnd] + ctx[:meanCondRest] / ctx[:nProps]
    return isfinite(v) ? v : 0.0
end

# Ablation components
sigmoid_shogenji(ctx) = begin
    cs = ctx[:prodP] > 1e-15 ? ctx[:pAnd] / ctx[:prodP] : 0.0
    cs / (cs + 1.0)
end

meancond_over_n(ctx) = ctx[:meanCondRest] / ctx[:nProps]

# ═══════════════════════════════════════════════════════════════════════════════
# All measures to evaluate
# ═══════════════════════════════════════════════════════════════════════════════

const ALL_MEASURES = [
    # Known measures from the literature
    ("Prior",           prior_measure),
    ("Shogenji",        shogenji),
    ("Olsson",          olsson),
    ("log-Shogenji",    log_shogenji),
    ("C_E",             ewing_measure),
    ("MeanCond",        mean_cond),
    ("C_d (D&M)",       cd_douven_meijs),
    ("O (Meijs)",       overlap_meijs),
    # Discovered measures
    ("D1:log-prod",     discovered_top1),
    ("D2:scaled-sig",   discovered_top2),
    ("D4:sig+P+MC/n",   discovered_top4),
    # Ablation: individual components of Top4
    ("sig(C_S)",        sigmoid_shogenji),
    ("MC/n",            meancond_over_n),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════════

function main(; seed=42)
    rng = MersenneTwister(seed)

    # ── Generate test data ──
    println("="^70)
    println("COHERENCE MEASURE EVALUATION")
    println("="^70)

    extended_n = [2, 3, 4, 5, 6, 7, 8]
    reliabilities = [0.3, 0.5, 0.7, 0.9]
    systems_per = [n <= 5 ? 300 : (n <= 7 ? 150 : 80) for n in extended_n]

    println("\nGenerating test data...")
    println("  n values: $extended_n")
    println("  Reliabilities: $reliabilities")
    println("  Systems per stratum: $systems_per")

    strata = Stratum[]
    for (ni, n_val) in enumerate(extended_n)
        for rel in reliabilities
            systems = [generate_system(rng, n_val; fixed_rel=rel)
                       for _ in 1:systems_per[ni]]
            ctxs = [s.ctx for s in systems]
            posts = Float64[s.posterior for s in systems]
            push!(strata, Stratum(n_val, rel, ctxs, posts))
        end
    end

    # Group by n and by reliability
    by_n = Dict{Int, Vector{Stratum}}()
    by_rel = Dict{Float64, Vector{Stratum}}()
    for s in strata
        push!(get!(by_n, s.n, Stratum[]), s)
        push!(get!(by_rel, s.reliability, Stratum[]), s)
    end

    total_systems = sum(length(s.ctxs) for s in strata)
    println("  Total systems generated: $total_systems\n")

    # Helper: compute a statistic per stratum, return averages grouped by key
    function eval_per_stratum(mfn, strata_list, stat_fn)
        vals = Float64[]
        for s in strata_list
            scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
            push!(vals, stat_fn(scores, s))
        end
        return mean(vals)
    end

    # ══════════════════════════════════════════════════════════════════════════
    # 1. RAW PEARSON r
    # ══════════════════════════════════════════════════════════════════════════

    println("="^70)
    println("1. RAW PEARSON r (standard evaluation)")
    println("="^70)

    # By n
    println("\n  Pearson r by n (averaged over reliabilities):\n")
    print("  $(rpad("Measure", 15))")
    for nv in extended_n; print(" │  n=$nv "); end
    println(" │  avg")
    print("  $(repeat("─", 15))")
    for _ in extended_n; print("─┼───────"); end
    println("─┼───────")

    for (name, mfn) in ALL_MEASURES
        print("  $(rpad(name, 15))")
        r_per_n = Float64[]
        for nv in extended_n
            avg = eval_per_stratum(mfn, by_n[nv],
                (scores, s) -> std(scores) > 1e-10 ? pearson_r(scores, s.posteriors) : 0.0)
            push!(r_per_n, avg)
            print(" │ $(lpad(round(avg, digits=3), 5)) ")
        end
        println(" │ $(lpad(round(mean(r_per_n), digits=3), 5))")
    end

    # By reliability
    println("\n  Pearson r by reliability (averaged over n):\n")
    rels_sorted = sort(collect(keys(by_rel)))
    print("  $(rpad("Measure", 15))")
    for r in rels_sorted; print(" │  r=$r "); end
    println(" │  avg")
    print("  $(repeat("─", 15))")
    for _ in rels_sorted; print("─┼────────"); end
    println("─┼───────")

    for (name, mfn) in ALL_MEASURES
        print("  $(rpad(name, 15))")
        r_per_rel = Float64[]
        for rel in rels_sorted
            avg = eval_per_stratum(mfn, by_rel[rel],
                (scores, s) -> std(scores) > 1e-10 ? pearson_r(scores, s.posteriors) : 0.0)
            push!(r_per_rel, avg)
            print(" │ $(lpad(round(avg, digits=3), 6)) ")
        end
        println(" │ $(lpad(round(mean(r_per_rel), digits=3), 5))")
    end

    # ══════════════════════════════════════════════════════════════════════════
    # 2. KENDALL τ-b
    # ══════════════════════════════════════════════════════════════════════════

    println("\n" * "="^70)
    println("2. KENDALL τ-b (for comparison with Angere 2008)")
    println("="^70)

    println("\n  Kendall τ-b by n (averaged over reliabilities):\n")
    print("  $(rpad("Measure", 15))")
    for nv in extended_n; print(" │  n=$nv "); end
    println(" │  avg")
    print("  $(repeat("─", 15))")
    for _ in extended_n; print("─┼───────"); end
    println("─┼───────")

    for (name, mfn) in ALL_MEASURES
        print("  $(rpad(name, 15))")
        τ_per_n = Float64[]
        for nv in extended_n
            avg = eval_per_stratum(mfn, by_n[nv],
                (scores, s) -> std(scores) > 1e-10 ? kendall_tau_b(scores, s.posteriors) : 0.5)
            push!(τ_per_n, avg)
            print(" │ $(lpad(round(avg, digits=3), 5)) ")
        end
        println(" │ $(lpad(round(mean(τ_per_n), digits=3), 5))")
    end

    # ══════════════════════════════════════════════════════════════════════════
    # 3. PARTIAL CORRELATION CONTROLLING FOR P(&A)
    # ══════════════════════════════════════════════════════════════════════════

    println("\n" * "="^70)
    println("3. PARTIAL PEARSON r CONTROLLING FOR P(&A)")
    println("   (Residual predictive power after removing prior's contribution)")
    println("="^70)

    # By n
    println("\n  Partial r by n (averaged over reliabilities):\n")
    print("  $(rpad("Measure", 15))")
    for nv in extended_n; print(" │  n=$nv "); end
    println(" │  avg")
    print("  $(repeat("─", 15))")
    for _ in extended_n; print("─┼───────"); end
    println("─┼───────")

    for (name, mfn) in ALL_MEASURES
        print("  $(rpad(name, 15))")
        pr_per_n = Float64[]
        for nv in extended_n
            avg = eval_per_stratum(mfn, by_n[nv], (scores, s) -> begin
                priors = Float64[ctx[:pAnd] for ctx in s.ctxs]
                partial_cor(scores, s.posteriors, priors)
            end)
            push!(pr_per_n, avg)
            print(" │ $(lpad(round(avg, digits=3), 5)) ")
        end
        println(" │ $(lpad(round(mean(pr_per_n), digits=3), 5))")
    end

    # By reliability
    println("\n  Partial r by reliability (averaged over n):\n")
    print("  $(rpad("Measure", 15))")
    for r in rels_sorted; print(" │  r=$r "); end
    println(" │  avg")
    print("  $(repeat("─", 15))")
    for _ in rels_sorted; print("─┼────────"); end
    println("─┼───────")

    for (name, mfn) in ALL_MEASURES
        print("  $(rpad(name, 15))")
        pr_per_rel = Float64[]
        for rel in rels_sorted
            avg = eval_per_stratum(mfn, by_rel[rel], (scores, s) -> begin
                priors = Float64[ctx[:pAnd] for ctx in s.ctxs]
                partial_cor(scores, s.posteriors, priors)
            end)
            push!(pr_per_rel, avg)
            print(" │ $(lpad(round(avg, digits=3), 6)) ")
        end
        println(" │ $(lpad(round(mean(pr_per_rel), digits=3), 5))")
    end

    # ══════════════════════════════════════════════════════════════════════════
    # 4. VARIANCE DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════════════

    println("\n" * "="^70)
    println("4. VARIANCE DECOMPOSITION: PRIOR vs COHERENCE")
    println("   R² of posterior ~ P(&A), posterior ~ C, posterior ~ P(&A) + C")
    println("="^70)

    println("\n  $(rpad("Measure", 15)) │ R²(prior) │  R²(C)  │ R²(both) │ ΔR²(C|pr) │  %ΔR²")
    println("  $(repeat("─", 15))─┼───────────┼─────────┼──────────┼───────────┼───────")

    for (name, mfn) in ALL_MEASURES
        r2_prior_all = Float64[]
        r2_C_all = Float64[]
        r2_both_all = Float64[]
        for s in strata
            scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
            priors = Float64[ctx[:pAnd] for ctx in s.ctxs]
            posts = s.posteriors

            r_py = std(priors) > 1e-10 ? pearson_r(priors, posts) : 0.0
            r_cy = std(scores) > 1e-10 ? pearson_r(scores, posts) : 0.0
            pr_cy = partial_cor(scores, posts, priors)

            push!(r2_prior_all, r_py^2)
            push!(r2_C_all, r_cy^2)
            push!(r2_both_all, 1.0 - (1.0 - r_py^2) * (1.0 - pr_cy^2))
        end
        r2p = mean(r2_prior_all)
        r2c = mean(r2_C_all)
        r2b = mean(r2_both_all)
        delta = r2b - r2p
        pct = r2p > 0 ? 100.0 * delta / r2p : 0.0
        println("  $(rpad(name, 15)) │ $(lpad(round(r2p, digits=4), 8))  │ $(lpad(round(r2c, digits=4), 6))  │ $(lpad(round(r2b, digits=4), 7))  │ $(lpad(round(delta, digits=4), 8))  │ $(lpad(round(pct, digits=1), 5))%")
    end

    # ══════════════════════════════════════════════════════════════════════════
    # 5. BOOTSTRAP CONFIDENCE INTERVALS FOR PARTIAL r
    # ══════════════════════════════════════════════════════════════════════════

    println("\n" * "="^70)
    println("5. BOOTSTRAP CI FOR PARTIAL r (B=2000)")
    println("="^70)

    B = 2000
    boot_rng = MersenneTwister(seed + 7777)

    println("\n  $(rpad("Measure", 15)) │  mean  │  [95% CI]           │   SE")
    println("  $(repeat("─", 15))─┼────────┼─────────────────────┼───────")

    for (name, mfn) in ALL_MEASURES
        # Compute partial r per stratum
        stratum_partial_r = Float64[]
        for s in strata
            scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
            priors = Float64[ctx[:pAnd] for ctx in s.ctxs]
            push!(stratum_partial_r, partial_cor(scores, s.posteriors, priors))
        end

        n_strata = length(stratum_partial_r)
        boot_means = Float64[]
        for _ in 1:B
            idx = [rand(boot_rng, 1:n_strata) for _ in 1:n_strata]
            push!(boot_means, mean(stratum_partial_r[i] for i in idx))
        end
        sort!(boot_means)
        ci_lo = boot_means[max(1, round(Int, 0.025 * B))]
        ci_hi = boot_means[min(B, round(Int, 0.975 * B))]
        m = mean(stratum_partial_r)
        se = std(boot_means)
        println("  $(rpad(name, 15)) │ $(lpad(round(m, digits=4), 6)) │  [$(round(ci_lo, digits=4)), $(round(ci_hi, digits=4))]  │ $(lpad(round(se, digits=4), 6))")
    end

    # ══════════════════════════════════════════════════════════════════════════
    # 6. PAIRWISE BOOTSTRAP TESTS FOR PARTIAL r
    # ══════════════════════════════════════════════════════════════════════════

    println("\n" * "="^70)
    println("6. PAIRWISE Δ(PARTIAL r) BOOTSTRAP TESTS")
    println("   (Discovered measures vs known coherence measures)")
    println("="^70)

    # Compute partial r per stratum for all measures
    partial_r_by_measure = Dict{String, Vector{Float64}}()
    for (name, mfn) in ALL_MEASURES
        pr_vals = Float64[]
        for s in strata
            scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
            priors = Float64[ctx[:pAnd] for ctx in s.ctxs]
            push!(pr_vals, partial_cor(scores, s.posteriors, priors))
        end
        partial_r_by_measure[name] = pr_vals
    end

    # Compare each discovered measure against key baselines
    discovered_names = ["D1:log-prod", "D2:scaled-sig", "D4:sig+P+MC/n"]
    baseline_names = ["Shogenji", "Olsson", "log-Shogenji", "C_E",
                      "MeanCond", "C_d (D&M)", "O (Meijs)",
                      "sig(C_S)", "MC/n"]

    println("\n  $(rpad("Comparison", 30)) │  Δr̄(partial) │  [95% CI]           │ p(Δr≤0)")
    println("  $(repeat("─", 30))─┼───────────────┼─────────────────────┼─────────")

    n_strata = length(strata)
    for dname in discovered_names
        d_vals = partial_r_by_measure[dname]
        for bname in baseline_names
            b_vals = partial_r_by_measure[bname]
            # Bootstrap Δr
            boot_deltas = Float64[]
            for _ in 1:B
                idx = [rand(boot_rng, 1:n_strata) for _ in 1:n_strata]
                d_boot = mean(d_vals[i] for i in idx)
                b_boot = mean(b_vals[i] for i in idx)
                push!(boot_deltas, d_boot - b_boot)
            end
            sort!(boot_deltas)
            ci_lo = boot_deltas[max(1, round(Int, 0.025 * B))]
            ci_hi = boot_deltas[min(B, round(Int, 0.975 * B))]
            delta_mean = mean(d_vals) - mean(b_vals)
            p_val = count(x -> x <= 0, boot_deltas) / B
            println("  $(rpad("$dname - $bname", 30)) │ $(lpad(round(delta_mean, digits=4), 12))  │ [$(round(ci_lo, digits=4)), $(round(ci_hi, digits=4))] │ $(round(p_val, digits=4))")
        end
        println()
    end

    # ══════════════════════════════════════════════════════════════════════════
    # 7. SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════

    println("="^70)
    println("7. SUMMARY: RAW r vs PARTIAL r (controlling for prior)")
    println("="^70)

    println("\n  $(rpad("Measure", 15)) │  raw r  │ partial r │   Δ   │ % retained")
    println("  $(repeat("─", 15))─┼─────────┼───────────┼───────┼───────────")

    for (name, mfn) in ALL_MEASURES
        raw_r_vals = Float64[]
        par_r_vals = Float64[]
        for s in strata
            scores = Float64[let v = mfn(ctx); isfinite(v) ? v : 0.0 end for ctx in s.ctxs]
            priors = Float64[ctx[:pAnd] for ctx in s.ctxs]
            raw = std(scores) > 1e-10 ? pearson_r(scores, s.posteriors) : 0.0
            par = partial_cor(scores, s.posteriors, priors)
            push!(raw_r_vals, raw)
            push!(par_r_vals, par)
        end
        raw_m = mean(raw_r_vals)
        par_m = mean(par_r_vals)
        delta = raw_m - par_m
        pct = raw_m > 0 ? 100.0 * par_m / raw_m : 0.0
        println("  $(rpad(name, 15)) │ $(lpad(round(raw_m, digits=4), 7)) │ $(lpad(round(par_m, digits=4), 9)) │ $(lpad(round(delta, digits=4), 5)) │ $(lpad(round(pct, digits=1), 7))%")
    end

    println("\n  Interpretation:")
    println("  - '% retained' shows how much predictive power survives after")
    println("    removing P(&A)'s contribution. Higher = more genuine coherence signal.")
    println("  - Prior should drop to ~0% (it IS the controlled variable).")
    println("  - Pure coherence measures (Shogenji, C_d, etc.) should retain most power.")
    println("  - Discovered measures: the key question is whether their advantage")
    println("    over MeanCond survives the prior-control test.")

    println("\nDone.")
end

main(seed=42)
