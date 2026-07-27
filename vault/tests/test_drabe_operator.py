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


def _operator(value: np.ndarray, gamma: float, *, buggy: bool) -> np.ndarray:
    continuation = np.minimum(
        _G, np.maximum(_L, value[_NEXT])
    )
    immediate = _G if buggy else np.minimum(_L, _G)
    return (1.0 - gamma) * immediate + gamma * continuation


def _fixed_point(gamma: float, *, buggy: bool) -> tuple[np.ndarray, list[float]]:
    value = np.zeros(4)
    exact = (
        np.array([1.0, 1.0, 1.0, -1.0])
        if buggy
        else np.array([1.0, 2.0 * gamma - 1.0, -1.0, -1.0])
    )
    errors = [float(np.max(np.abs(value - exact)))]
    for _ in range(100_000):
        value = _operator(value, gamma, buggy=buggy)
        errors.append(float(np.max(np.abs(value - exact))))
        if errors[-1] < 1e-12:
            break
    return value, errors


@pytest.mark.parametrize("gamma", [0.9, 0.99, 0.999])
def test_corrected_drabe_is_contractive_and_conservative(
    gamma: float,
) -> None:
    discounted, errors = _fixed_point(gamma, buggy=False)

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
def test_old_immediate_term_has_a_false_safe_trap(gamma: float) -> None:
    buggy, _ = _fixed_point(gamma, buggy=True)
    buggy_set = set(np.flatnonzero(buggy >= 0.0))
    undiscounted_set = {0, 1}

    assert buggy[2] == pytest.approx(1.0, abs=1e-11)
    assert 2 in buggy_set
    assert 2 not in undiscounted_set
    assert not buggy_set <= undiscounted_set


def test_torch_target_uses_minimum_immediate_term_and_terminal_case() -> None:
    gs = torch.tensor([[1.0], [1.0]])
    ls = torch.tensor([[-1.0], [-1.0]])
    next_q = torch.tensor([[-1.0], [0.7]])
    dones = torch.tensor([[0.0], [1.0]])

    actual = _drabe_target_torch(gs, ls, next_q, dones, gamma=0.9)

    torch.testing.assert_close(actual, torch.tensor([[-1.0], [-1.0]]))
