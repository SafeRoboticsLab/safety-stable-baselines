"""Formal regression tests for the positive-good DRABE operator."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from vault.reach_avoid_sac import _drabe_target_torch


# States: target, can-reach-target, safe-but-trapped, failed.
_G = np.array([1.0, 1.0, 1.0, -1.0])
_L = np.array([1.0, -1.0, -1.0, -1.0])
_NEXT = np.array([0, 0, 2, 3])


def _production_operator(value: np.ndarray, gamma: float) -> np.ndarray:
    """Evaluate the shipped torch backup on the deterministic toy MDP."""
    result = _drabe_target_torch(
        torch.tensor(_G[:, None], dtype=torch.float64),
        torch.tensor(_L[:, None], dtype=torch.float64),
        torch.tensor(value[_NEXT, None], dtype=torch.float64),
        torch.zeros((len(_G), 1), dtype=torch.float64),
        gamma,
    )
    return result.numpy().reshape(-1)


def _avoid_only_immediate_operator(
    value: np.ndarray, gamma: float
) -> np.ndarray:
    """Use the valid avoid-only immediate term in an invalid reach-avoid mix."""
    continuation = np.minimum(_G, np.maximum(_L, value[_NEXT]))
    return (1.0 - gamma) * _G + gamma * continuation


def _fixed_point(
    gamma: float, operator
) -> tuple[np.ndarray, list[float]]:
    value = np.zeros(4)
    exact = np.array([1.0, 2.0 * gamma - 1.0, -1.0, -1.0])
    errors = [float(np.max(np.abs(value - exact)))]
    for _ in range(100_000):
        value = operator(value, gamma)
        errors.append(float(np.max(np.abs(value - exact))))
        if errors[-1] < 1e-12:
            break
    return value, errors


@pytest.mark.parametrize("gamma", [0.9, 0.99, 0.999])
def test_corrected_drabe_is_contractive_and_conservative(
    gamma: float,
) -> None:
    discounted, errors = _fixed_point(gamma, _production_operator)

    # Finite-horizon RABE value iteration from its one-step payoff.
    undiscounted = np.minimum(_L, _G)
    for _ in range(8):
        undiscounted = np.minimum(
            _G, np.maximum(_L, undiscounted[_NEXT])
        )
    np.testing.assert_array_equal(undiscounted, [1.0, 1.0, -1.0, -1.0])
    np.testing.assert_allclose(
        discounted,
        [1.0, 2.0 * gamma - 1.0, -1.0, -1.0],
        atol=1e-11,
    )

    discounted_set = set(np.flatnonzero(discounted >= 0.0))
    undiscounted_set = set(np.flatnonzero(undiscounted >= 0.0))
    assert discounted_set <= undiscounted_set

    # Banach bound from the gamma-contraction.
    for iteration, error in enumerate(errors[:200]):
        assert error <= gamma**iteration * errors[0] + 1e-14


@pytest.mark.parametrize("gamma", [0.9, 0.99, 0.999])
def test_avoid_only_immediate_term_is_not_a_reach_avoid_backup(
    gamma: float,
) -> None:
    """The upstream avoid-only term is correct, but not for reach-avoid."""
    value = np.zeros(4)
    for _ in range(100_000):
        updated = _avoid_only_immediate_operator(value, gamma)
        if np.max(np.abs(updated - value)) < 1e-12:
            value = updated
            break
        value = updated
    avoid_only_set = set(np.flatnonzero(value >= 0.0))
    undiscounted_set = {0, 1}

    assert value[2] == pytest.approx(1.0, abs=2e-9)
    assert 2 in avoid_only_set
    assert 2 not in undiscounted_set
    assert not avoid_only_set <= undiscounted_set


def test_torch_target_observes_reach_avoid_structure_and_gamma() -> None:
    gs = torch.tensor([[0.8], [0.8], [0.8]])
    ls = torch.tensor([[-0.2], [-0.2], [-0.2]])
    next_q = torch.tensor([[0.6], [1.4], [1.4]])
    dones = torch.tensor([[0.0], [0.0], [1.0]])

    actual = _drabe_target_torch(gs, ls, next_q, dones, gamma=0.9)

    # Case 1 makes max(ls, next_q) observable; case 2 makes the outer
    # min(gs, ...) observable; case 3 pins the terminal immediate term.
    torch.testing.assert_close(
        actual,
        torch.tensor([[0.52], [0.70], [-0.20]]),
    )
