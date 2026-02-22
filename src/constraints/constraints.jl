#=
Constraints Module
==================

Provides constraint checking for symbolic optimization problems.
Constraints can be used to ensure discovered formulas satisfy
theoretical requirements beyond just optimizing objectives.

Two modes:
- Hard constraints: Violating individuals are rejected
- Soft constraints: Violations added as penalty to objectives

Example use cases:
- Confirmation measures: directionality, logicality, symmetry
- Probability aggregators: boundary conditions, monotonicity
- Scoring rules: properness conditions
=#

module Constraints

using Statistics: mean
using Random: MersenneTwister

export Constraint, ConstraintSet
export check_constraint, check_constraints, violation_rate
export directionality_constraint, logicality_constraint, symmetry_constraint
export monotonicity_constraint, boundary_constraint

# ─────────────────────────────────────────────────────────────────────────────
# Constraint Types
# ─────────────────────────────────────────────────────────────────────────────

"""
    Constraint

A constraint that candidate expressions must satisfy.

Fields:
- `name`: Identifier for the constraint
- `test_fn`: Function `(tree, evaluate_fn) -> (satisfied::Bool, violation_score::Float64)`
- `description`: Human-readable description
"""
struct Constraint
    name::Symbol
    test_fn::Function
    description::String
end

"""
    ConstraintSet

A collection of constraints to check together.
"""
struct ConstraintSet
    constraints::Vector{Constraint}
    mode::Symbol  # :hard or :soft
    penalty_weight::Float64  # For soft mode
end

ConstraintSet(constraints::Vector{Constraint}; mode::Symbol=:soft, penalty_weight::Float64=1.0) = 
    ConstraintSet(constraints, mode, penalty_weight)

# ─────────────────────────────────────────────────────────────────────────────
# Constraint Checking
# ─────────────────────────────────────────────────────────────────────────────

"""
    check_constraint(c::Constraint, tree, evaluate_fn) -> (satisfied, violation_score)

Check if a tree satisfies a constraint.
Returns (true, 0.0) if satisfied, (false, score) if violated.
"""
check_constraint(c::Constraint, tree, evaluate_fn) = c.test_fn(tree, evaluate_fn)

"""
    check_constraints(cs::ConstraintSet, tree, evaluate_fn) -> (all_satisfied, total_penalty, details)

Check all constraints in a set.
Returns overall satisfaction, total penalty score, and per-constraint details.
"""
function check_constraints(cs::ConstraintSet, tree, evaluate_fn)
    details = Dict{Symbol, Tuple{Bool, Float64}}()
    total_penalty = 0.0
    all_satisfied = true
    
    for c in cs.constraints
        satisfied, violation = check_constraint(c, tree, evaluate_fn)
        details[c.name] = (satisfied, violation)
        if !satisfied
            all_satisfied = false
            total_penalty += violation * cs.penalty_weight
        end
    end
    
    return (all_satisfied, total_penalty, details)
end

"""
    violation_rate(c::Constraint, tree, evaluate_fn, test_cases) -> Float64

Compute the fraction of test cases where the constraint is violated.
"""
function violation_rate(c::Constraint, tree, evaluate_fn, test_cases)
    violations = 0
    for case in test_cases
        satisfied, _ = c.test_fn(tree, x -> evaluate_fn(x, case))
        if !satisfied
            violations += 1
        end
    end
    return violations / length(test_cases)
end

# ─────────────────────────────────────────────────────────────────────────────
# Confirmation Theory Constraints
# ─────────────────────────────────────────────────────────────────────────────

"""
    directionality_constraint(; n_tests=100, seed=42) -> Constraint

Confirmation measure should be:
- Positive when P(H|E) > P(H) (E confirms H)
- Negative when P(H|E) < P(H) (E disconfirms H)
- Zero when P(H|E) = P(H) (E is irrelevant to H)

Uses probabilistic test cases to check this property.
"""
function directionality_constraint(; n_tests::Int=100, seed::Int=42)
    Constraint(
        :directionality,
        (tree, evaluate_fn) -> begin
            rng = MersenneTwister(seed)
            
            violations = 0
            total = 0
            
            for _ in 1:n_tests
                # Generate random probabilities
                pH = rand(rng) * 0.8 + 0.1  # Avoid extremes
                pE = rand(rng) * 0.8 + 0.1
                
                # Generate pH_E that's either > pH, < pH, or ≈ pH
                case_type = rand(rng, 1:3)
                if case_type == 1  # Confirming: P(H|E) > P(H)
                    pH_E = pH + rand(rng) * (1 - pH) * 0.8
                    expected_sign = 1
                elseif case_type == 2  # Disconfirming: P(H|E) < P(H)
                    pH_E = pH * rand(rng) * 0.8
                    expected_sign = -1
                else  # Neutral: P(H|E) ≈ P(H)
                    pH_E = pH + (rand(rng) - 0.5) * 0.05
                    expected_sign = 0
                end
                
                pH_E = clamp(pH_E, 0.01, 0.99)
                
                # Derive other probabilities (simplified, assuming independence structure)
                pnotH = 1 - pH
                pnotE = 1 - pE
                pE_H = pH_E * pE / pH
                pE_H = clamp(pE_H, 0.01, 0.99)
                pE_notH = (pE - pH * pE_H) / pnotH
                pE_notH = clamp(pE_notH, 0.01, 0.99)
                pH_notE = (pH - pH_E * pE) / pnotE
                pH_notE = clamp(pH_notE, 0.01, 0.99)
                pnotH_E = 1 - pH_E
                pnotE_H = 1 - pE_H
                pnotH_notE = 1 - pH_notE
                pnotE_notH = 1 - pE_notH
                pH_and_E = pH_E * pE
                
                ctx = Dict{Symbol, Any}(
                    :pH => pH, :pE => pE, :pnotH => pnotH, :pnotE => pnotE,
                    :pH_E => pH_E, :pE_H => pE_H, :pH_notE => pH_notE, :pE_notH => pE_notH,
                    :pnotH_E => pnotH_E, :pnotE_H => pnotE_H,
                    :pnotH_notE => pnotH_notE, :pnotE_notH => pnotE_notH,
                    :pH_and_E => pH_and_E
                )
                
                score = evaluate_fn(tree, ctx)
                
                if isfinite(score)
                    total += 1
                    actual_sign = sign(score)
                    
                    # Check directionality
                    if expected_sign == 1 && actual_sign <= 0
                        violations += 1
                    elseif expected_sign == -1 && actual_sign >= 0
                        violations += 1
                    # For neutral cases, we're lenient (small values either way ok)
                    end
                end
            end
            
            violation_rate = total > 0 ? violations / total : 1.0
            satisfied = violation_rate < 0.1  # Allow 10% tolerance
            
            return (satisfied, violation_rate)
        end,
        "Positive when E confirms H, negative when E disconfirms H"
    )
end

"""
    logicality_constraint(; n_tests=50, seed=42) -> Constraint

Confirmation measure should achieve:
- Maximum value when E logically entails H (E ⊨ H, so P(H|E) = 1)
- Minimum value when E logically entails ¬H (E ⊨ ¬H, so P(H|E) = 0)
"""
function logicality_constraint(; n_tests::Int=50, seed::Int=42)
    Constraint(
        :logicality,
        (tree, evaluate_fn) -> begin
            rng = MersenneTwister(seed)
            
            entailment_scores = Float64[]
            contradiction_scores = Float64[]
            regular_scores = Float64[]
            
            for _ in 1:n_tests
                pH = rand(rng) * 0.6 + 0.2
                pE = rand(rng) * 0.6 + 0.2
                pnotH = 1 - pH
                pnotE = 1 - pE
                
                # Case 1: E entails H (P(H|E) = 1)
                ctx_entails = Dict{Symbol, Any}(
                    :pH => pH, :pE => pE, :pnotH => pnotH, :pnotE => pnotE,
                    :pH_E => 1.0, :pE_H => pE / pH,
                    :pH_notE => (pH - pE) / pnotE,
                    :pE_notH => 0.0,  # E → H means ¬H → ¬E
                    :pnotH_E => 0.0,
                    :pnotE_H => 1 - pE / pH,
                    :pnotH_notE => pnotH / pnotE,
                    :pnotE_notH => 1.0,
                    :pH_and_E => pE
                )
                
                # Clamp values
                for k in keys(ctx_entails)
                    if ctx_entails[k] isa Float64
                        ctx_entails[k] = clamp(ctx_entails[k], 0.001, 0.999)
                    end
                end
                ctx_entails[:pH_E] = 1.0  # Keep this at exactly 1
                ctx_entails[:pE_notH] = 0.001  # Near zero
                ctx_entails[:pnotH_E] = 0.001
                
                score = evaluate_fn(tree, ctx_entails)
                if isfinite(score)
                    push!(entailment_scores, score)
                end
                
                # Case 2: E entails ¬H (P(H|E) = 0)
                ctx_contradicts = Dict{Symbol, Any}(
                    :pH => pH, :pE => pE, :pnotH => pnotH, :pnotE => pnotE,
                    :pH_E => 0.001,  # Near zero
                    :pE_H => 0.001,
                    :pH_notE => pH / pnotE,
                    :pE_notH => pE / pnotH,
                    :pnotH_E => 1.0,
                    :pnotE_H => 1.0,
                    :pnotH_notE => (pnotH - pE) / pnotE,
                    :pnotE_notH => 1 - pE / pnotH,
                    :pH_and_E => 0.001
                )
                
                for k in keys(ctx_contradicts)
                    if ctx_contradicts[k] isa Float64
                        ctx_contradicts[k] = clamp(ctx_contradicts[k], 0.001, 0.999)
                    end
                end
                ctx_contradicts[:pH_E] = 0.001
                ctx_contradicts[:pnotH_E] = 0.999
                
                score = evaluate_fn(tree, ctx_contradicts)
                if isfinite(score)
                    push!(contradiction_scores, score)
                end
                
                # Case 3: Regular case for comparison
                pH_E_reg = rand(rng) * 0.6 + 0.2
                ctx_regular = Dict{Symbol, Any}(
                    :pH => pH, :pE => pE, :pnotH => pnotH, :pnotE => pnotE,
                    :pH_E => pH_E_reg, :pE_H => pH_E_reg * pE / pH,
                    :pH_notE => 0.5, :pE_notH => 0.5,
                    :pnotH_E => 1 - pH_E_reg, :pnotE_H => 0.5,
                    :pnotH_notE => 0.5, :pnotE_notH => 0.5,
                    :pH_and_E => pH_E_reg * pE
                )
                
                score = evaluate_fn(tree, ctx_regular)
                if isfinite(score)
                    push!(regular_scores, score)
                end
            end
            
            # Check: entailment scores should be higher than regular and contradiction
            # Contradiction scores should be lower than regular and entailment
            if isempty(entailment_scores) || isempty(contradiction_scores) || isempty(regular_scores)
                return (false, 1.0)
            end
            
            mean_entail = mean(entailment_scores)
            mean_contra = mean(contradiction_scores)
            mean_regular = mean(regular_scores)
            
            # Entailment should give highest scores, contradiction lowest
            entail_ok = mean_entail > mean_regular
            contra_ok = mean_contra < mean_regular
            ordering_ok = mean_entail > mean_contra
            
            violations = (entail_ok ? 0 : 1) + (contra_ok ? 0 : 1) + (ordering_ok ? 0 : 1)
            satisfied = violations == 0
            
            return (satisfied, violations / 3.0)
        end,
        "Maximum when E⊨H, minimum when E⊨¬H"
    )
end

"""
    symmetry_constraint(; type=:equivalence, n_tests=50, seed=42) -> Constraint

Symmetry constraints for confirmation measures:
- `:equivalence` - C(H,E) = C(¬H,¬E) (Eells-Fitelson symmetry)
- `:sign` - sign(C(H,E)) = -sign(C(¬H,E)) 
- `:commutativity` - C(H,E) = C(E,H) (controversial, often rejected)
"""
function symmetry_constraint(; type::Symbol=:equivalence, n_tests::Int=50, seed::Int=42)
    name = Symbol("symmetry_", type)
    
    description = if type == :equivalence
        "C(H,E) = C(¬H,¬E)"
    elseif type == :sign
        "sign(C(H,E)) = -sign(C(¬H,E))"
    elseif type == :commutativity
        "C(H,E) = C(E,H)"
    else
        "Unknown symmetry type"
    end
    
    Constraint(
        name,
        (tree, evaluate_fn) -> begin
            rng = MersenneTwister(seed)
            
            violations = 0
            total = 0
            
            for _ in 1:n_tests
                # Generate a random probability setup
                pH = rand(rng) * 0.6 + 0.2
                pE = rand(rng) * 0.6 + 0.2
                pH_E = rand(rng) * 0.6 + 0.2
                
                pnotH = 1 - pH
                pnotE = 1 - pE
                
                # Derive consistent probabilities
                pH_and_E = pH_E * pE
                pH_and_E = clamp(pH_and_E, 0.01, min(pH, pE) - 0.01)
                
                pE_H = pH_and_E / pH
                pH_notE = (pH - pH_and_E) / pnotE
                pE_notH = (pE - pH_and_E) / pnotH
                pnotH_E = 1 - pH_E
                pnotE_H = 1 - pE_H
                pnotH_notE = (pnotH - (pE - pH_and_E)) / pnotE
                pnotE_notH = 1 - pE_notH
                
                # Clamp all
                pE_H = clamp(pE_H, 0.01, 0.99)
                pH_notE = clamp(pH_notE, 0.01, 0.99)
                pE_notH = clamp(pE_notH, 0.01, 0.99)
                pnotH_notE = clamp(pnotH_notE, 0.01, 0.99)
                pnotE_notH = clamp(pnotE_notH, 0.01, 0.99)
                
                ctx = Dict{Symbol, Any}(
                    :pH => pH, :pE => pE, :pnotH => pnotH, :pnotE => pnotE,
                    :pH_E => pH_E, :pE_H => pE_H, :pH_notE => pH_notE, :pE_notH => pE_notH,
                    :pnotH_E => pnotH_E, :pnotE_H => pnotE_H,
                    :pnotH_notE => pnotH_notE, :pnotE_notH => pnotE_notH,
                    :pH_and_E => pH_and_E
                )
                
                score_HE = evaluate_fn(tree, ctx)
                
                if type == :equivalence
                    # C(H,E) should equal C(¬H,¬E)
                    # For C(¬H,¬E), swap H↔¬H and E↔¬E
                    ctx_swap = Dict{Symbol, Any}(
                        :pH => pnotH, :pE => pnotE, :pnotH => pH, :pnotE => pE,
                        :pH_E => pnotH_notE, :pE_H => pnotE_notH,
                        :pH_notE => pnotH_E, :pE_notH => pnotE_H,
                        :pnotH_E => pH_notE, :pnotE_H => pE_H,
                        :pnotH_notE => pH_E, :pnotE_notH => pE_notH,
                        :pH_and_E => pnotH_notE * pnotE
                    )
                    score_notHnotE = evaluate_fn(tree, ctx_swap)
                    
                    if isfinite(score_HE) && isfinite(score_notHnotE)
                        total += 1
                        if abs(score_HE - score_notHnotE) > 0.1 * max(abs(score_HE), abs(score_notHnotE), 0.1)
                            violations += 1
                        end
                    end
                    
                elseif type == :sign
                    # sign(C(H,E)) = -sign(C(¬H,E))
                    ctx_notH = Dict{Symbol, Any}(
                        :pH => pnotH, :pE => pE, :pnotH => pH, :pnotE => pnotE,
                        :pH_E => pnotH_E, :pE_H => pE_notH,
                        :pH_notE => pnotH_notE, :pE_notH => pE_H,
                        :pnotH_E => pH_E, :pnotE_H => pnotE_notH,
                        :pnotH_notE => pH_notE, :pnotE_notH => pnotE_H,
                        :pH_and_E => pnotH_E * pE
                    )
                    score_notHE = evaluate_fn(tree, ctx_notH)
                    
                    if isfinite(score_HE) && isfinite(score_notHE)
                        total += 1
                        if sign(score_HE) != -sign(score_notHE) && 
                           abs(score_HE) > 0.05 && abs(score_notHE) > 0.05
                            violations += 1
                        end
                    end
                end
            end
            
            violation_rate = total > 0 ? violations / total : 1.0
            satisfied = violation_rate < 0.15  # 15% tolerance
            
            return (satisfied, violation_rate)
        end,
        description
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# General-Purpose Constraints
# ─────────────────────────────────────────────────────────────────────────────

"""
    monotonicity_constraint(; variable, direction=:increasing, n_tests=50, seed=42)

Formula should be monotonic in the specified variable.
"""
function monotonicity_constraint(; variable::Symbol, direction::Symbol=:increasing, 
                                  n_tests::Int=50, seed::Int=42)
    Constraint(
        Symbol("monotonic_", variable),
        (tree, evaluate_fn) -> begin
            rng = MersenneTwister(seed)
            
            violations = 0
            total = 0
            
            for _ in 1:n_tests
                # Create base context
                base_ctx = Dict{Symbol, Any}()
                for v in [:pH, :pE, :pnotH, :pnotE, :pH_E, :pE_H, :pH_notE, :pE_notH,
                          :pnotH_E, :pnotE_H, :pnotH_notE, :pnotE_notH, :pH_and_E]
                    base_ctx[v] = rand(rng) * 0.8 + 0.1
                end
                
                # Test monotonicity
                lo_ctx = copy(base_ctx)
                hi_ctx = copy(base_ctx)
                lo_ctx[variable] = 0.2
                hi_ctx[variable] = 0.8
                
                score_lo = evaluate_fn(tree, lo_ctx)
                score_hi = evaluate_fn(tree, hi_ctx)
                
                if isfinite(score_lo) && isfinite(score_hi)
                    total += 1
                    if direction == :increasing && score_hi < score_lo
                        violations += 1
                    elseif direction == :decreasing && score_hi > score_lo
                        violations += 1
                    end
                end
            end
            
            violation_rate = total > 0 ? violations / total : 1.0
            satisfied = violation_rate < 0.1
            
            return (satisfied, violation_rate)
        end,
        "$(direction == :increasing ? "Increasing" : "Decreasing") in $variable"
    )
end

"""
    boundary_constraint(; conditions, n_tests=50, seed=42)

Formula should satisfy boundary conditions.
`conditions` is a vector of (input_dict, expected_output) pairs.
"""
function boundary_constraint(; conditions::Vector, tolerance::Float64=0.1)
    Constraint(
        :boundary,
        (tree, evaluate_fn) -> begin
            violations = 0
            
            for (inputs, expected) in conditions
                ctx = Dict{Symbol, Any}(k => v for (k, v) in inputs)
                score = evaluate_fn(tree, ctx)
                
                if isfinite(score) && isfinite(expected)
                    if abs(score - expected) > tolerance
                        violations += 1
                    end
                elseif isfinite(score) != isfinite(expected)
                    violations += 1
                end
            end
            
            violation_rate = violations / length(conditions)
            satisfied = violation_rate < 0.1
            
            return (satisfied, violation_rate)
        end,
        "Boundary conditions"
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Pre-built Constraint Sets
# ─────────────────────────────────────────────────────────────────────────────

"""
    confirmation_measure_constraints(; mode=:soft, penalty_weight=0.5) -> ConstraintSet

Standard constraints for confirmation measures:
- Directionality
- Logicality  
- Equivalence symmetry
"""
function confirmation_measure_constraints(; mode::Symbol=:soft, penalty_weight::Float64=0.5)
    ConstraintSet(
        [
            directionality_constraint(),
            logicality_constraint(),
            symmetry_constraint(type=:equivalence)
        ],
        mode=mode,
        penalty_weight=penalty_weight
    )
end

export confirmation_measure_constraints

end # module Constraints
