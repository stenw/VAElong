"""
Survival-analysis utilities for the joint longitudinal-survival model.

The helpers in this module are intentionally model-agnostic. They implement
the numerical building blocks described in Section 3.2 of
``DLVMs_LongData_v7.pdf`` without changing any longitudinal-only behavior.
"""

import math

import torch


_GK15_NODES = torch.tensor(
    [
        -0.9914553711208126,
        -0.9491079123427585,
        -0.8648644233597691,
        -0.7415311855993945,
        -0.5860872354676911,
        -0.4058451513773972,
        -0.2077849550078985,
        0.0,
        0.2077849550078985,
        0.4058451513773972,
        0.5860872354676911,
        0.7415311855993945,
        0.8648644233597691,
        0.9491079123427585,
        0.9914553711208126,
    ],
    dtype=torch.float32,
)

_GK15_WEIGHTS = torch.tensor(
    [
        0.0229353220105292,
        0.0630920926299786,
        0.1047900103222502,
        0.1406532597155259,
        0.1690047266392679,
        0.1903505780647854,
        0.2044329400752989,
        0.2094821410847278,
        0.2044329400752989,
        0.1903505780647854,
        0.1690047266392679,
        0.1406532597155259,
        0.1047900103222502,
        0.0630920926299786,
        0.0229353220105292,
    ],
    dtype=torch.float32,
)


def gauss_kronrod_15(device=None, dtype=None):
    """Return 15-point Gauss-Kronrod nodes and weights on ``[-1, 1]``."""
    if dtype is None:
        dtype = torch.float32
    return (
        _GK15_NODES.to(device=device, dtype=dtype),
        _GK15_WEIGHTS.to(device=device, dtype=dtype),
    )


def rescale_quadrature(nodes, weights, lower, upper):
    """Rescale quadrature nodes/weights from ``[-1, 1]`` to ``[lower, upper]``.

    Args:
        nodes: 1D tensor of quadrature nodes on ``[-1, 1]``.
        weights: 1D tensor of matching quadrature weights.
        lower: Scalar or tensor broadcastable to the batch shape.
        upper: Scalar or tensor broadcastable to the batch shape.

    Returns:
        scaled_nodes: Tensor of shape ``(..., Q)``.
        scaled_weights: Tensor of shape ``(..., Q)``.
    """
    nodes = torch.as_tensor(nodes)
    weights = torch.as_tensor(weights, device=nodes.device, dtype=nodes.dtype)
    lower = torch.as_tensor(lower, device=nodes.device, dtype=nodes.dtype)
    upper = torch.as_tensor(upper, device=nodes.device, dtype=nodes.dtype)

    half_width = 0.5 * (upper - lower)
    center = 0.5 * (upper + lower)
    scaled_nodes = center.unsqueeze(-1) + half_width.unsqueeze(-1) * nodes
    scaled_weights = half_width.unsqueeze(-1) * weights
    return scaled_nodes, scaled_weights


def subject_quadrature_grid(event_times, lower=None, device=None, dtype=None):
    """Return subject-specific quadrature nodes/weights on ``[lower, event_times]``."""
    event_times = torch.as_tensor(event_times, device=device, dtype=dtype or torch.float32)
    if lower is None:
        lower = torch.zeros_like(event_times)
    nodes, weights = gauss_kronrod_15(device=event_times.device, dtype=event_times.dtype)
    return rescale_quadrature(nodes, weights, lower=lower, upper=event_times)


def compose_log_hazard(log_baseline_hazard, log_relative_risk):
    """Compose the log hazard from baseline and subject-specific terms."""
    return log_baseline_hazard + log_relative_risk


def cumulative_hazard_from_hazard(hazard_values, weights):
    """Approximate the cumulative hazard from hazard evaluations and weights."""
    hazard_values = torch.as_tensor(hazard_values)
    weights = torch.as_tensor(weights, device=hazard_values.device, dtype=hazard_values.dtype)
    return (hazard_values * weights).sum(dim=-1)


def cumulative_hazard_from_log_terms(log_terms, max_cumulative_hazard=1e12):
    """Approximate ``sum(exp(log_terms))`` with a stable log-sum-exp routine."""
    log_terms = torch.as_tensor(log_terms)
    max_term = log_terms.amax(dim=-1, keepdim=True)
    shifted = torch.exp(log_terms - max_term)
    tiny = torch.finfo(log_terms.dtype).tiny
    log_total = max_term.squeeze(-1) + torch.log(shifted.sum(dim=-1).clamp(min=tiny))
    max_log_total = math.log(float(max_cumulative_hazard))
    total = torch.exp(torch.clamp(log_total, max=max_log_total))
    return torch.clamp(total, max=float(max_cumulative_hazard))


def cumulative_hazard_from_log_hazard(log_hazard_values, weights,
                                      max_cumulative_hazard=1e12):
    """Approximate ``int h(s) ds`` from log-hazard evaluations and weights."""
    log_hazard_values = torch.as_tensor(log_hazard_values)
    weights = torch.as_tensor(
        weights, device=log_hazard_values.device, dtype=log_hazard_values.dtype
    )
    tiny = torch.finfo(log_hazard_values.dtype).tiny
    log_terms = torch.log(weights.clamp(min=tiny)) + log_hazard_values
    return cumulative_hazard_from_log_terms(
        log_terms,
        max_cumulative_hazard=max_cumulative_hazard,
    )


def survival_log_likelihood(log_hazard_event, cumulative_hazard, event_indicator):
    """Return ``delta * log h(T) - H(T)`` for right-censored survival data."""
    log_hazard_event = torch.as_tensor(log_hazard_event)
    cumulative_hazard = torch.as_tensor(
        cumulative_hazard,
        device=log_hazard_event.device,
        dtype=log_hazard_event.dtype,
    )
    event_indicator = torch.as_tensor(
        event_indicator,
        device=log_hazard_event.device,
        dtype=log_hazard_event.dtype,
    )
    return event_indicator * log_hazard_event - cumulative_hazard


def log_survival_probability(cumulative_hazard):
    """Return ``log S(t) = -H(t)``."""
    cumulative_hazard = torch.as_tensor(cumulative_hazard)
    return -cumulative_hazard


def survival_probability(cumulative_hazard, min_log_survival=-80.0):
    """Return ``S(t) = exp(-H(t))`` with mild underflow protection."""
    log_survival = log_survival_probability(cumulative_hazard)
    return torch.exp(torch.clamp(log_survival, min=min_log_survival, max=0.0))


__all__ = [
    "gauss_kronrod_15",
    "rescale_quadrature",
    "subject_quadrature_grid",
    "compose_log_hazard",
    "cumulative_hazard_from_hazard",
    "cumulative_hazard_from_log_terms",
    "cumulative_hazard_from_log_hazard",
    "survival_log_likelihood",
    "log_survival_probability",
    "survival_probability",
]
