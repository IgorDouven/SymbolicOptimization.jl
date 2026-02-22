# ═══════════════════════════════════════════════════════════════════════════════
# Symbolics.jl Integration Interface
# ═══════════════════════════════════════════════════════════════════════════════
#
# These functions are stubs that get real implementations when the user loads
# Symbolics.jl (via a package extension in ext/SymbolicsExt.jl).
#
# Usage:
#   using SymbolicOptimization
#   using Symbolics  # triggers the extension
#
#   tree = FunctionNode(:+, FunctionNode(:*, Variable(:x), Variable(:x)), Constant(-1.0))
#   deep_simplify(tree)           # → simplified AbstractNode tree
#   simplified_string(tree)       # → "x^2 - 1"
#   simplified_latex(tree)        # → "x^{2} - 1"

const _SYMBOLICS_NOT_LOADED_MSG = """
This function requires Symbolics.jl. Load it with:
    using Symbolics
Then retry your call."""

"""
    to_symbolics(tree::AbstractNode) -> Symbolics.Num

Convert an `AbstractNode` expression tree into a `Symbolics.Num` expression.

Safe operators (e.g., `safe_div`, `safe_log`) are mapped to their standard
mathematical equivalents so that Symbolics.jl can reason about them.

Requires `using Symbolics`.
"""
function to_symbolics end

"""
    from_symbolics(expr) -> AbstractNode

Convert a `Symbolics.Num` expression back into an `AbstractNode` tree.

Standard mathematical operators are preserved (not mapped to safe variants),
so the resulting tree is best used for display or analysis rather than
inside the GP loop.

Requires `using Symbolics`.
"""
function from_symbolics end

"""
    deep_simplify(tree::AbstractNode; expand::Bool=false) -> AbstractNode

Perform deep algebraic simplification using Symbolics.jl.

Converts the tree to a Symbolics expression, applies `Symbolics.simplify`,
and converts back to an `AbstractNode`. This can resolve identities like
`(x + 1) * (x - 1) → x^2 - 1` that the built-in `simplify` cannot.

# Keyword Arguments
- `expand::Bool=false`: If `true`, also expand products into sums.

Requires `using Symbolics`.

# Example
```julia
using Symbolics
tree = FunctionNode(:*, 
    FunctionNode(:+, Variable(:x), Constant(1.0)),
    FunctionNode(:-, Variable(:x), Constant(1.0)))
deep_simplify(tree)  # x^2 - 1
```
"""
function deep_simplify end

"""
    simplified_string(tree::AbstractNode; digits::Int=3) -> String

Return a simplified human-readable string representation of the tree
using Symbolics.jl for algebraic simplification.

Requires `using Symbolics`.
"""
function simplified_string end

"""
    simplified_latex(tree::AbstractNode) -> String

Return a simplified LaTeX representation of the tree using Symbolics.jl.

Requires `using Symbolics`.
"""
function simplified_latex end

"""
    PiecewiseResult

Result of `simplify_piecewise`, containing the separated and simplified branches.

# Fields
- `condition::AbstractNode`: The condition tree (from inside `step_func(...)`)
- `condition_string::String`: Human-readable condition
- `if_branch::AbstractNode`: Simplified tree for the `condition ≥ 0` case
- `else_branch::AbstractNode`: Simplified tree for the `condition < 0` case
- `if_string::String`: Simplified string for the if-branch
- `else_string::String`: Simplified string for the else-branch
- `if_latex::String`: LaTeX for the if-branch
- `else_latex::String`: LaTeX for the else-branch
"""
struct PiecewiseResult
    condition::AbstractNode
    condition_string::String
    if_branch::AbstractNode
    else_branch::AbstractNode
    if_string::String
    else_string::String
    if_latex::String
    else_latex::String
end

function Base.show(io::IO, r::PiecewiseResult)
    println(io, "Piecewise formula:")
    println(io, "  When $(r.condition_string) ≥ 0:")
    println(io, "    $(r.if_string)")
    println(io, "  When $(r.condition_string) < 0:")
    print(io, "    $(r.else_string)")
end

"""
    simplify_piecewise(tree::AbstractNode; indicator::Symbol=:step_func,
                       substitutions::Dict{Symbol,AbstractNode}=Dict{Symbol,AbstractNode}()) -> PiecewiseResult

Simplify a piecewise formula by separating branches and simplifying each independently.

Detects `indicator(cond) * A + (1 - indicator(cond)) * B` patterns (in any tree shape),
substitutes `indicator(cond) = 1` and `indicator(cond) = 0` to extract branches,
then uses `deep_simplify` on each branch.

# Keyword Arguments
- `indicator::Symbol=:step_func`: Name of the step/indicator function in the tree.
- `substitutions::Dict{Symbol,AbstractNode}`: Variable identities to apply before
  simplifying, e.g. `Dict(:pnotH_notE => FunctionNode(:-, Constant(1.0), Variable(:pH_notE)))`.
  This lets Symbolics.jl exploit domain constraints like `P(¬H|¬E) = 1 - P(H|¬E)`.
  See also `complement_vars` for a convenient way to build these.

Requires `using Symbolics`.

# Example
```julia
using Symbolics

# With domain substitutions via complement_vars helper
result = simplify_piecewise(tree, substitutions = complement_vars(
    :pnotH_notE => :pH_notE,
    :pnotH_E    => :pH_E,
))
println(result)
# Piecewise formula:
#   When (pH_E - pH_notE) ≥ 0:
#     (pH_E - pH_notE) * pH_notE / (1 - pH_notE)
#   When (pH_E - pH_notE) < 0:
#     (pH_E - pH_notE) * (1 - pH_notE) / pH_notE
```
"""
function simplify_piecewise end

"""
    complement_vars(pairs::Pair{Symbol,Symbol}...) -> Dict{Symbol, AbstractNode}

Build a substitution dictionary for complementary probability variables.

Each pair `a => b` creates the substitution `a = 1 - b`. This is useful for
telling `simplify_piecewise` about identities like `P(¬H|¬E) = 1 - P(H|¬E)`.

# Example
```julia
subs = complement_vars(:pnotH_notE => :pH_notE, :pnotH_E => :pH_E)
# Equivalent to:
# Dict(:pnotH_notE => FunctionNode(:-, Constant(1.0), Variable(:pH_notE)),
#      :pnotH_E    => FunctionNode(:-, Constant(1.0), Variable(:pH_E)))

result = simplify_piecewise(tree, substitutions = subs)
```
"""
function complement_vars(pairs::Pair{Symbol,Symbol}...)
    Dict{Symbol, AbstractNode}(
        k => FunctionNode(:-, Constant(1.0), Variable(v))
        for (k, v) in pairs
    )
end
