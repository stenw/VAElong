"""
Variable type configuration for mixed-type longitudinal data.
"""

"""
    VariableSpec

Specification for a single time-varying variable.

# Fields
- `name::String`: Name of the variable
- `var_type::String`: One of "continuous", "binary", "bounded"
- `lower::Float32`: Lower bound (only used for "bounded" type)
- `upper::Float32`: Upper bound (only used for "bounded" type)
- `index::Int`: 1-based index in the feature dimension (set automatically by VariableConfig)
"""
struct VariableSpec
    name::String
    var_type::String
    lower::Float32
    upper::Float32
    index::Int

    function VariableSpec(name::String, var_type::String;
                          lower::Float32=0.0f0, upper::Float32=1.0f0, index::Int=0)
        if !(var_type in ("continuous", "binary", "bounded"))
            error("var_type must be 'continuous', 'binary', or 'bounded', got '$var_type'")
        end
        if var_type == "bounded" && lower >= upper
            error("For bounded variables, lower ($lower) must be less than upper ($upper)")
        end
        new(name, var_type, lower, upper, index)
    end
end

"""
    VariableConfig

Configuration for all variables in the longitudinal data.

# Fields
- `variables::Vector{VariableSpec}`: List of variable specifications (indices set to 1:n)
"""
struct VariableConfig
    variables::Vector{VariableSpec}

    function VariableConfig(variables::Vector{VariableSpec})
        # Re-create each VariableSpec with correct 1-based index
        indexed = [VariableSpec(v.name, v.var_type; lower=v.lower, upper=v.upper, index=i)
                   for (i, v) in enumerate(variables)]
        new(indexed)
    end
end

"""Number of features."""
n_features(vc::VariableConfig) = length(vc.variables)

"""Indices (1-based) of continuous variables."""
function continuous_indices(vc::VariableConfig)
    return [v.index for v in vc.variables if v.var_type == "continuous"]
end

"""Indices (1-based) of binary variables."""
function binary_indices(vc::VariableConfig)
    return [v.index for v in vc.variables if v.var_type == "binary"]
end

"""Indices (1-based) of bounded variables."""
function bounded_indices(vc::VariableConfig)
    return [v.index for v in vc.variables if v.var_type == "bounded"]
end

"""Return Dict{Int, Tuple{Float32, Float32}} of (lower, upper) for bounded variables."""
function get_bounds(vc::VariableConfig)
    return Dict(v.index => (v.lower, v.upper)
                for v in vc.variables if v.var_type == "bounded")
end

"""Create a VariableConfig where all variables are continuous (backward compatible)."""
function all_continuous(n::Int)
    variables = [VariableSpec("feature_$i", "continuous") for i in 1:n]
    return VariableConfig(variables)
end
