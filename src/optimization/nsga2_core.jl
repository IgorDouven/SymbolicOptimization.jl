# ═══════════════════════════════════════════════════════════════════════════════
# NSGA-II Core Algorithms
# ═══════════════════════════════════════════════════════════════════════════════

using Random

# ───────────────────────────────────────────────────────────────────────────────
# Pareto Dominance
# ───────────────────────────────────────────────────────────────────────────────

"""
    dominates(a::Individual, b::Individual, minimize::Vector{Bool}) -> Bool

Check if individual `a` dominates individual `b`.
`a` dominates `b` if `a` is at least as good in all objectives and strictly better in at least one.
"""
function dominates(a::Individual, b::Individual, minimize::Vector{Bool})
    dominated_in_any = false
    better_in_at_least_one = false
    
    for (i, (oa, ob)) in enumerate(zip(a.objectives, b.objectives))
        if minimize[i]
            # Lower is better
            if oa > ob
                return false  # a is worse in this objective
            elseif oa < ob
                better_in_at_least_one = true
            end
        else
            # Higher is better
            if oa < ob
                return false  # a is worse in this objective
            elseif oa > ob
                better_in_at_least_one = true
            end
        end
    end
    
    return better_in_at_least_one
end

"""
    dominates(a::Vector{Float64}, b::Vector{Float64}, minimize::Vector{Bool}) -> Bool

Check if objective vector `a` dominates objective vector `b`.
"""
function dominates(a::Vector{Float64}, b::Vector{Float64}, minimize::Vector{Bool})
    better_in_at_least_one = false
    
    for (i, (oa, ob)) in enumerate(zip(a, b))
        if minimize[i]
            if oa > ob
                return false
            elseif oa < ob
                better_in_at_least_one = true
            end
        else
            if oa < ob
                return false
            elseif oa > ob
                better_in_at_least_one = true
            end
        end
    end
    
    return better_in_at_least_one
end

# ───────────────────────────────────────────────────────────────────────────────
# Non-Dominated Sorting
# ───────────────────────────────────────────────────────────────────────────────

"""
    nondominated_sort!(population::Vector{Individual}, objectives::Vector{ObjectiveFunction})

Perform non-dominated sorting on the population, assigning ranks to each individual.
Rank 1 = Pareto front (non-dominated), Rank 2 = dominated only by rank 1, etc.

Modifies individuals in place by setting their `rank` field.
Returns a vector of fronts (each front is a vector of indices).
"""
function nondominated_sort!(population::Vector{Individual}, objectives::Vector{ObjectiveFunction})
    n = length(population)
    minimize = [obj.minimize for obj in objectives]
    
    # For each individual, track:
    # - domination_count: how many individuals dominate this one
    # - dominated_set: which individuals this one dominates
    domination_count = zeros(Int, n)
    dominated_sets = [Int[] for _ in 1:n]
    
    # First front (Pareto front)
    fronts = Vector{Vector{Int}}()
    front1 = Int[]
    
    # Compare all pairs
    for i in 1:n
        for j in 1:n
            if i != j
                if dominates(population[i], population[j], minimize)
                    push!(dominated_sets[i], j)
                elseif dominates(population[j], population[i], minimize)
                    domination_count[i] += 1
                end
            end
        end
        
        # If not dominated by anyone, it's in the first front
        if domination_count[i] == 0
            population[i].rank = 1
            push!(front1, i)
        end
    end
    
    push!(fronts, front1)
    
    # Generate subsequent fronts
    current_front = 1
    while current_front <= length(fronts) && !isempty(fronts[current_front])
        next_front = Int[]
        
        for i in fronts[current_front]
            for j in dominated_sets[i]
                domination_count[j] -= 1
                if domination_count[j] == 0
                    population[j].rank = current_front + 1
                    push!(next_front, j)
                end
            end
        end
        
        if !isempty(next_front)
            push!(fronts, next_front)
        end
        current_front += 1
        
        # Safety check
        if current_front > n
            break
        end
    end
    
    return fronts
end

"""
    get_pareto_front_from_pop(population::Vector{Individual}, objectives::Vector{ObjectiveFunction}) -> Vector{Individual}

Extract the Pareto front (rank 1 individuals) from a population.
Does not modify the population.
"""
function get_pareto_front_from_pop(population::Vector{Individual}, objectives::Vector{ObjectiveFunction})
    # Make a copy to avoid modifying original
    pop_copy = [copy(ind) for ind in population]
    fronts = nondominated_sort!(pop_copy, objectives)
    
    if isempty(fronts) || isempty(fronts[1])
        return Individual[]
    end
    
    return [pop_copy[i] for i in fronts[1]]
end

# ───────────────────────────────────────────────────────────────────────────────
# Crowding Distance
# ───────────────────────────────────────────────────────────────────────────────

"""
    compute_crowding_distance!(population::Vector{Individual}, front_indices::Vector{Int})

Compute crowding distance for individuals in a front.
Individuals at the boundaries get infinite distance.
Others get distance based on the hypervolume of the cuboid formed by their neighbors.

Modifies individuals in place by setting their `crowding_distance` field.
"""
function compute_crowding_distance!(population::Vector{Individual}, front_indices::Vector{Int})
    n = length(front_indices)
    
    if n == 0
        return
    end
    
    # Reset crowding distances
    for i in front_indices
        population[i].crowding_distance = 0.0
    end
    
    if n <= 2
        # All boundary points get infinite distance
        for i in front_indices
            population[i].crowding_distance = Inf
        end
        return
    end
    
    n_objectives = length(population[front_indices[1]].objectives)
    
    # For each objective
    for m in 1:n_objectives
        # Sort by this objective
        sorted_indices = sort(front_indices, by=i -> population[i].objectives[m])
        
        # Boundary points get infinite distance
        population[sorted_indices[1]].crowding_distance = Inf
        population[sorted_indices[end]].crowding_distance = Inf
        
        # Calculate range for normalization
        obj_min = population[sorted_indices[1]].objectives[m]
        obj_max = population[sorted_indices[end]].objectives[m]
        range_m = obj_max - obj_min
        
        if range_m < 1e-10
            continue  # All same value, skip
        end
        
        # Calculate crowding distance for intermediate points
        for i in 2:(n-1)
            idx = sorted_indices[i]
            if !isinf(population[idx].crowding_distance)
                prev_obj = population[sorted_indices[i-1]].objectives[m]
                next_obj = population[sorted_indices[i+1]].objectives[m]
                population[idx].crowding_distance += (next_obj - prev_obj) / range_m
            end
        end
    end
end

"""
    compute_all_crowding_distances!(population::Vector{Individual}, fronts::Vector{Vector{Int}})

Compute crowding distances for all fronts.
"""
function compute_all_crowding_distances!(population::Vector{Individual}, fronts::Vector{Vector{Int}})
    for front in fronts
        compute_crowding_distance!(population, front)
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Selection
# ───────────────────────────────────────────────────────────────────────────────

"""
    crowded_comparison(a::Individual, b::Individual) -> Bool

Compare two individuals using crowded comparison operator.
Returns true if `a` is better than `b`.
Prefers lower rank, and for same rank, prefers higher crowding distance.
"""
function crowded_comparison(a::Individual, b::Individual)
    if a.rank < b.rank
        return true
    elseif a.rank > b.rank
        return false
    else
        # Same rank - prefer higher crowding distance
        return a.crowding_distance > b.crowding_distance
    end
end

"""
    parsimony_comparison(a::Individual, b::Individual, tolerance::Float64, complexity_idx::Int) -> Bool

Compare two individuals with parsimony pressure.
Returns true if `a` is better than `b`.

If the primary objective (index 1) differs by less than `tolerance` (as a fraction),
prefer the solution with lower complexity (at `complexity_idx`).
Otherwise, fall back to standard crowded comparison.

IMPORTANT: Solutions with Inf/NaN primary objective are always considered worse.
"""
function parsimony_comparison(a::Individual, b::Individual, tolerance::Float64, complexity_idx::Int=3)
    # Handle Inf/NaN - always prefer finite over non-finite
    a_finite = !isempty(a.objectives) && isfinite(a.objectives[1])
    b_finite = !isempty(b.objectives) && isfinite(b.objectives[1])
    
    if a_finite && !b_finite
        return true   # a is finite, b is not -> a wins
    elseif !a_finite && b_finite
        return false  # b is finite, a is not -> b wins
    elseif !a_finite && !b_finite
        # Both non-finite - fall back to crowding
        return a.crowding_distance > b.crowding_distance
    end
    
    # Both finite - proceed with normal comparison
    
    # First compare by rank
    if a.rank < b.rank
        return true
    elseif a.rank > b.rank
        return false
    end
    
    # Same rank - check if primary objectives are within tolerance
    if tolerance > 0
        obj_a = a.objectives[1]
        obj_b = b.objectives[1]
        
        # Compute relative difference
        avg = (abs(obj_a) + abs(obj_b)) / 2
        if avg > 0
            rel_diff = abs(obj_a - obj_b) / avg
            
            if rel_diff < tolerance
                # Within tolerance - prefer simpler solution
                if complexity_idx <= length(a.objectives) && complexity_idx <= length(b.objectives)
                    comp_a = a.objectives[complexity_idx]
                    comp_b = b.objectives[complexity_idx]
                    if comp_a != comp_b
                        return comp_a < comp_b  # Lower complexity is better
                    end
                end
            end
        end
    end
    
    # Fall back to crowding distance
    return a.crowding_distance > b.crowding_distance
end

"""
    tournament_select(population::Vector{Individual}, tournament_size::Int; 
                      rng=Random.GLOBAL_RNG, parsimony_tolerance=0.0, complexity_idx=3) -> Individual

Select an individual using tournament selection with crowded comparison.
If `parsimony_tolerance > 0`, applies parsimony pressure to prefer simpler solutions
when primary objectives are within the tolerance.
"""
function tournament_select(population::Vector{Individual}, tournament_size::Int;
                           rng::AbstractRNG=Random.GLOBAL_RNG,
                           parsimony_tolerance::Float64=0.0,
                           complexity_idx::Int=3)
    # Select random competitors
    competitors = rand(rng, population, tournament_size)
    
    # Find the best using appropriate comparison
    best = competitors[1]
    for i in 2:tournament_size
        challenger = competitors[i]
        is_better = if parsimony_tolerance > 0
            parsimony_comparison(challenger, best, parsimony_tolerance, complexity_idx)
        else
            crowded_comparison(challenger, best)
        end
        if is_better
            best = challenger
        end
    end
    
    return best
end

"""
    select_parents(population::Vector{Individual}, n::Int, tournament_size::Int; rng=Random.GLOBAL_RNG) -> Vector{Individual}

Select n parents using tournament selection.
"""
function select_parents(population::Vector{Individual}, n::Int, tournament_size::Int;
                        rng::AbstractRNG=Random.GLOBAL_RNG)
    [tournament_select(population, tournament_size; rng=rng) for _ in 1:n]
end

# ───────────────────────────────────────────────────────────────────────────────
# Environmental Selection
# ───────────────────────────────────────────────────────────────────────────────

"""
    environmental_select!(combined::Vector{Individual}, target_size::Int, 
                          objectives::Vector{ObjectiveFunction}) -> Vector{Individual}

Select the best individuals to survive to the next generation.
Uses non-dominated sorting and crowding distance.
"""
function environmental_select!(combined::Vector{Individual}, target_size::Int,
                               objectives::Vector{ObjectiveFunction})
    if length(combined) <= target_size
        return combined
    end
    
    # Perform non-dominated sorting
    fronts = nondominated_sort!(combined, objectives)
    
    # Compute crowding distances
    compute_all_crowding_distances!(combined, fronts)
    
    # Select individuals front by front
    selected = Individual[]
    
    for front in fronts
        if length(selected) + length(front) <= target_size
            # Add entire front
            for i in front
                push!(selected, combined[i])
            end
        else
            # Need to select from this front based on crowding distance
            remaining_slots = target_size - length(selected)
            
            # Sort front by crowding distance (descending)
            front_sorted = sort(front, by=i -> combined[i].crowding_distance, rev=true)
            
            for i in 1:remaining_slots
                push!(selected, combined[front_sorted[i]])
            end
            
            break
        end
    end
    
    return selected
end
