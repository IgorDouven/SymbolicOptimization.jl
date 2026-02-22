# ═══════════════════════════════════════════════════════════════════════════════
# Safe Operator Implementations
# ═══════════════════════════════════════════════════════════════════════════════
#
# Following SymbolicRegression.jl's approach: operators must be defined over
# the entire real line and return NaN where not defined. This prevents crashes
# during evolutionary search.

# ───────────────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────────────

const SAFE_EPS = 1e-10
const SAFE_MAX_EXP = 20.0
const SAFE_MIN_LOG = 1e-10

# ───────────────────────────────────────────────────────────────────────────────
# Scalar Safe Operations
# ───────────────────────────────────────────────────────────────────────────────

safe_div(x, y) = abs(y) < SAFE_EPS ? NaN : x / y

function safe_pow(x, y)
    y_clamped = clamp(y, -SAFE_MAX_EXP, SAFE_MAX_EXP)
    if x < 0 && !isinteger(y_clamped)
        return NaN
    end
    result = x^y_clamped
    return isfinite(result) ? result : NaN
end

safe_log(x) = x > SAFE_MIN_LOG ? log(x) : NaN
safe_log10(x) = x > SAFE_MIN_LOG ? log10(x) : NaN
safe_log2(x) = x > SAFE_MIN_LOG ? log2(x) : NaN
safe_exp(x) = exp(clamp(x, -SAFE_MAX_EXP, SAFE_MAX_EXP))
safe_sqrt(x) = x >= 0 ? sqrt(x) : NaN
safe_inv(x) = abs(x) < SAFE_EPS ? NaN : 1.0 / x

function safe_tan(x)
    result = tan(x)
    return isfinite(result) ? result : NaN
end

safe_asin(x) = asin(clamp(x, -1.0, 1.0))
safe_acos(x) = acos(clamp(x, -1.0, 1.0))

safe_sinh(x) = sinh(clamp(x, -SAFE_MAX_EXP, SAFE_MAX_EXP))
safe_cosh(x) = cosh(clamp(x, -SAFE_MAX_EXP, SAFE_MAX_EXP))
safe_tanh(x) = tanh(x)

safe_acosh(x) = x >= 1.0 ? acosh(x) : NaN
safe_atanh(x) = abs(x) < 1.0 ? atanh(x) : NaN

# ───────────────────────────────────────────────────────────────────────────────
# Element-wise Safe Operations (for vectors)
# ───────────────────────────────────────────────────────────────────────────────

ew_safe_div(x, y) = safe_div.(x, y)
ew_safe_pow(x, y) = safe_pow.(x, y)
ew_safe_log(x) = safe_log.(x)
ew_safe_exp(x) = safe_exp.(x)
ew_safe_sqrt(x) = safe_sqrt.(x)

# Vector with scalar exponent (common case)
function ew_pow_scalar(x::AbstractVector, y::Real)
    y_clamped = clamp(y, -SAFE_MAX_EXP, SAFE_MAX_EXP)
    return [safe_pow(xi, y_clamped) for xi in x]
end

# ───────────────────────────────────────────────────────────────────────────────
# Reduction Operations (Vector → Scalar)
# ───────────────────────────────────────────────────────────────────────────────

safe_mean(x) = isempty(x) ? NaN : Statistics.mean(x)
safe_sum(x) = isempty(x) ? 0.0 : sum(x)
safe_prod(x) = isempty(x) ? 1.0 : (r = prod(x); isfinite(r) ? r : NaN)
safe_std(x) = length(x) < 2 ? 0.0 : Statistics.std(x)
safe_var(x) = length(x) < 2 ? 0.0 : Statistics.var(x)
safe_maximum(x) = isempty(x) ? NaN : maximum(x)
safe_minimum(x) = isempty(x) ? NaN : minimum(x)
safe_median(x) = isempty(x) ? NaN : Statistics.median(x)

# ───────────────────────────────────────────────────────────────────────────────
# Special Functions
# ───────────────────────────────────────────────────────────────────────────────

sigmoid(x) = 1.0 / (1.0 + safe_exp(-x))
relu(x) = max(0.0, x)
softplus(x) = log(1.0 + safe_exp(x))
clamp01(x) = clamp(x, 0.0, 1.0)
gaussian(x) = safe_exp(-x^2)
step_func(x) = x >= 0.0 ? 1.0 : 0.0

# ───────────────────────────────────────────────────────────────────────────────
# Conditional Operators
# ───────────────────────────────────────────────────────────────────────────────
#
# These enable the discovery of piecewise functions like Crupi's z measure:
#   z = (P(H|E) - P(H)) / (1 - P(H))    if P(H|E) ≥ P(H)
#   z = (P(H|E) - P(H)) / P(H)          if P(H|E) < P(H)
#
# With these primitives, such functions can be constructed:
#   ifelse(P(H|E) - P(H), ..., ...)  -- branches based on sign of condition
#   step(x) * A + (1 - step(x)) * B  -- equivalent explicit construction

"""
Heaviside step function: 1 if x ≥ 0, else 0.
Use with arithmetic to construct conditionals:
    if cond ≥ 0 then A else B = A * step(cond) + B * (1 - step(cond))
"""
safe_step(x) = x >= 0.0 ? 1.0 : 0.0

"""
Sign function: 1 if x > 0, -1 if x < 0, 0 if x = 0.
Useful for signed conditionals.
"""
safe_sign(x) = x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0)

"""
Absolute value. Safe for all inputs.
Note: abs(x) = x * sign(x) = step(x) * x - step(-x) * x
"""
safe_abs(x) = abs(x)

"""
Positive part: max(x, 0). Returns x if x > 0, else 0.
"""
safe_pos(x) = max(x, 0.0)

"""
Negative part: min(x, 0). Returns x if x < 0, else 0.
Note: x = pos(x) + neg(x)
"""
safe_neg(x) = min(x, 0.0)

"""
Safe maximum of two values.
"""
safe_max(x, y) = max(x, y)

"""
Safe minimum of two values.
"""
safe_min(x, y) = min(x, y)

"""
Ternary conditional: if condition ≥ 0 then a else b.

This is the KEY operator for discovering piecewise functions.
The condition being ≥ 0 (rather than > 0) matches mathematical conventions
and the Heaviside step function.

Examples:
    ifelse(P(H|E) - P(H), A, B)  →  A if confirming, B if disconfirming
    ifelse(x, x, -x)             →  abs(x)
    ifelse(x - y, x, y)          →  max(x, y)
"""
function safe_ifelse(condition, then_val, else_val)
    return condition >= 0.0 ? then_val : else_val
end

"""
Smoothed/soft conditional using sigmoid. Differentiable approximation to ifelse.
Parameter k controls sharpness (higher = sharper transition).
"""
function soft_ifelse(condition, then_val, else_val; k=10.0)
    s = sigmoid(k * condition)
    return s * then_val + (1 - s) * else_val
end

# ───────────────────────────────────────────────────────────────────────────────
# Safe Operator Registry
# ───────────────────────────────────────────────────────────────────────────────

"""
Maps operator symbols/functions to their safe implementations.
"""
const SAFE_IMPLEMENTATIONS = Dict{Any, Function}(
    # Binary arithmetic
    (+) => +,
    (-) => -,
    (*) => *,
    (/) => safe_div,
    (^) => safe_pow,
    :+ => +,
    :- => -,
    :* => *,
    :/ => safe_div,
    :^ => safe_pow,
    
    # Unary arithmetic
    :neg => -,
    :abs => safe_abs,
    :sign => safe_sign,
    :inv => safe_inv,
    :square => x -> x^2,
    :cube => x -> x^3,
    
    # Trigonometric
    sin => sin,
    cos => cos,
    tan => safe_tan,
    :sin => sin,
    :cos => cos,
    :tan => safe_tan,
    :asin => safe_asin,
    :acos => safe_acos,
    :atan => atan,
    
    # Hyperbolic
    :sinh => safe_sinh,
    :cosh => safe_cosh,
    :tanh => safe_tanh,
    :asinh => asinh,
    :acosh => safe_acosh,
    :atanh => safe_atanh,
    
    # Exponential/Log
    exp => safe_exp,
    log => safe_log,
    sqrt => safe_sqrt,
    :exp => safe_exp,
    :log => safe_log,
    :log10 => safe_log10,
    :log2 => safe_log2,
    :sqrt => safe_sqrt,
    :cbrt => cbrt,
    
    # Special
    :sigmoid => sigmoid,
    :relu => relu,
    :softplus => softplus,
    :clamp01 => clamp01,
    :gaussian => gaussian,
    :step => safe_step,
    
    # Conditional operators (KEY for piecewise functions)
    :step_func => safe_step,  # Alias
    :pos => safe_pos,
    :neg_part => safe_neg,  # Note: :neg is already negation
    abs => safe_abs,
    sign => safe_sign,
    
    # Binary special
    max => safe_max,
    min => safe_min,
    :max => safe_max,
    :min => safe_min,
    :mod => (x, y) -> abs(y) < SAFE_EPS ? NaN : mod(x, y),
    :hypot => hypot,
    
    # Ternary conditional (KEY operator)
    :ifelse => safe_ifelse,
    :soft_ifelse => soft_ifelse,
    
    # Reductions (Vector → Scalar)
    :mean => safe_mean,
    :sum => safe_sum,
    :prod => safe_prod,
    :std => safe_std,
    :var => safe_var,
    :maximum => safe_maximum,
    :minimum => safe_minimum,
    :median => safe_median,
    :length => x -> Float64(length(x)),
    
    # Element-wise
    Symbol(".+") => .+,
    Symbol(".-") => .-,
    Symbol(".*") => .*,
    Symbol("./") => ew_safe_div,
    Symbol(".^") => ew_safe_pow,
    :ew_log => ew_safe_log,
    :ew_exp => ew_safe_exp,
    :ew_sqrt => ew_safe_sqrt,
)

"""
Default arity for known operators.
"""
const DEFAULT_ARITY = Dict{Any, Int}(
    # Binary
    (+) => 2, (-) => 2, (*) => 2, (/) => 2, (^) => 2,
    :+ => 2, :- => 2, :* => 2, :/ => 2, :^ => 2,
    max => 2, min => 2, :max => 2, :min => 2, :mod => 2, :hypot => 2,
    Symbol(".+") => 2, Symbol(".-") => 2, Symbol(".*") => 2,
    Symbol("./") => 2, Symbol(".^") => 2,
    
    # Unary
    sin => 1, cos => 1, tan => 1, exp => 1, log => 1, sqrt => 1,
    :sin => 1, :cos => 1, :tan => 1, :exp => 1, :log => 1, :sqrt => 1,
    :neg => 1, :abs => 1, :sign => 1, :inv => 1, :square => 1, :cube => 1,
    :asin => 1, :acos => 1, :atan => 1,
    :sinh => 1, :cosh => 1, :tanh => 1, :asinh => 1, :acosh => 1, :atanh => 1,
    :log10 => 1, :log2 => 1, :cbrt => 1,
    :sigmoid => 1, :relu => 1, :softplus => 1, :clamp01 => 1, :gaussian => 1, :step => 1,
    :mean => 1, :sum => 1, :prod => 1, :std => 1, :var => 1,
    :maximum => 1, :minimum => 1, :median => 1, :length => 1,
    :ew_log => 1, :ew_exp => 1, :ew_sqrt => 1,
    
    # Conditional unary
    :step_func => 1, :pos => 1, :neg_part => 1,
    abs => 1, sign => 1,
    
    # Ternary
    :ifelse => 3, :soft_ifelse => 3,
)

"""
    get_safe_version(func, arity::Int) -> Function

Get a safe version of the function that handles edge cases gracefully.
"""
function _get_safe_version(func, arity::Int)
    # Check registry first
    if haskey(SAFE_IMPLEMENTATIONS, func)
        return SAFE_IMPLEMENTATIONS[func]
    end
    
    # For symbols, check by symbol
    if func isa Symbol && haskey(SAFE_IMPLEMENTATIONS, func)
        return SAFE_IMPLEMENTATIONS[func]
    end
    
    # For functions, check by function object
    if func isa Function && haskey(SAFE_IMPLEMENTATIONS, func)
        return SAFE_IMPLEMENTATIONS[func]
    end
    
    # Fall back to wrapping in try-catch
    return _wrap_safe(func, arity)
end

function _wrap_safe(f::Function, arity::Int)
    if arity == 1
        return function(x)
            try
                result = f(x)
                return _validate_result(result)
            catch
                return NaN
            end
        end
    elseif arity == 2
        return function(x, y)
            try
                result = f(x, y)
                return _validate_result(result)
            catch
                return NaN
            end
        end
    else
        return function(args...)
            try
                result = f(args...)
                return _validate_result(result)
            catch
                return NaN
            end
        end
    end
end

"""
Validate a result, returning it if valid or NaN if not.
Handles both scalar and vector outputs.
"""
function _validate_result(result)
    if result isa Number
        return isfinite(result) ? result : NaN
    elseif result isa AbstractVector
        # For vectors, return as-is (individual elements may be NaN but that's handled elsewhere)
        return result
    elseif result isa AbstractMatrix
        return result
    else
        # Unknown type - return as-is and let caller handle
        return result
    end
end

"""
    get_default_arity(func) -> Union{Int, Nothing}

Get the default arity for a known operator, or nothing if unknown.
"""
function _get_default_arity(func)
    if haskey(DEFAULT_ARITY, func)
        return DEFAULT_ARITY[func]
    end
    return nothing
end
