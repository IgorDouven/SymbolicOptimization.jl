# Genetic Operators

SymbolicOptimization provides operators for generating, mutating, and recombining expression trees.

## Tree Generation

```julia
tree = generate_tree(grammar; method=GrowMethod(), max_depth=4)
pop = generate_population(grammar, 100; method=HalfAndHalfMethod())
```

Generation methods:
- [`FullMethod`](@ref) — all branches reach maximum depth
- [`GrowMethod`](@ref) — branches terminate randomly
- [`HalfAndHalfMethod`](@ref) — mix of full and grow

## Mutation

```julia
mutated = mutate(tree, grammar)               # Auto-select type
mutated = mutate_point(tree, grammar)         # Swap operator/variable
mutated = mutate_subtree(tree, grammar)       # Replace subtree
mutated = mutate_hoist(tree)                  # Promote subtree to root
mutated = mutate_constants(tree; std=0.5)     # Perturb constants
mutated = mutate_single_constant(tree)        # Perturb one constant
mutated = mutate_insert(tree, grammar)        # Add operator
mutated = mutate_delete(tree)                 # Remove operator
```

## Crossover

```julia
child1, child2 = crossover(parent1, parent2)                # Subtree exchange
child1, child2 = crossover_subtree(parent1, parent2)        # Same as above
child1, child2 = crossover_one_point(parent1, parent2)      # One-point
child1, child2 = crossover_uniform(parent1, parent2)        # Uniform
child1, child2 = crossover_size_fair(parent1, parent2)      # Size-fair
```

## Simplification

```julia
simplified = simplify(tree; grammar=grammar)   # Constant folding + algebra
simplified = simplify_algebra(tree)            # x + 0 -> x, x * 1 -> x, etc.
simplified = simplify_constants(tree)          # 1 + 2 -> 3
normalized = normalize_constants(tree)         # Round near-integer constants
clamped = clamp_constants(tree)                # Clamp extreme values
```

For deep algebraic simplification, see [Advanced Topics](@ref "Advanced Topics").

## Operators Reference

```@docs
GenerationMethod
FullMethod
GrowMethod
HalfAndHalfMethod
generate_tree
generate_population
random_terminal
generate_random_subtree
mutate
mutate_point
mutate_subtree
mutate_hoist
mutate_constants
mutate_single_constant
mutate_insert
mutate_delete
crossover
crossover_subtree
crossover_one_point
crossover_uniform
crossover_size_fair
simplify
simplify_algebra
simplify_constants
normalize_constants
clamp_constants
```
