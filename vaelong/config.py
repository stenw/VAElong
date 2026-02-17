"""
Variable type configuration for mixed-type longitudinal data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class VariableSpec:
    """Specification for a single time-varying variable.

    Args:
        name: Name of the variable
        var_type: One of 'continuous', 'binary', 'bounded'
        lower: Lower bound (only used for 'bounded' type)
        upper: Upper bound (only used for 'bounded' type)
    """
    name: str
    var_type: str
    lower: float = 0.0
    upper: float = 1.0
    index: Optional[int] = None

    def __post_init__(self):
        if self.var_type not in ('continuous', 'binary', 'bounded'):
            raise ValueError(
                f"var_type must be 'continuous', 'binary', or 'bounded', got '{self.var_type}'"
            )
        if self.var_type == 'bounded' and self.lower >= self.upper:
            raise ValueError(
                f"For bounded variables, lower ({self.lower}) must be less than upper ({self.upper})"
            )


@dataclass
class VariableConfig:
    """Configuration for all variables in the longitudinal data.

    Args:
        variables: List of VariableSpec instances
    """
    variables: List[VariableSpec]

    def __post_init__(self):
        for i, var in enumerate(self.variables):
            var.index = i

    @property
    def n_features(self) -> int:
        return len(self.variables)

    @property
    def continuous_indices(self) -> List[int]:
        return [v.index for v in self.variables if v.var_type == 'continuous']

    @property
    def binary_indices(self) -> List[int]:
        return [v.index for v in self.variables if v.var_type == 'binary']

    @property
    def bounded_indices(self) -> List[int]:
        return [v.index for v in self.variables if v.var_type == 'bounded']

    def get_bounds(self) -> Dict[int, Tuple[float, float]]:
        """Return {index: (lower, upper)} for bounded variables."""
        return {v.index: (v.lower, v.upper) for v in self.variables if v.var_type == 'bounded'}

    @classmethod
    def all_continuous(cls, n_features: int) -> 'VariableConfig':
        """Create config where all variables are continuous (backward compat)."""
        variables = [
            VariableSpec(name=f'feature_{i}', var_type='continuous')
            for i in range(n_features)
        ]
        return cls(variables=variables)
