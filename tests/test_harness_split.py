"""Acceptance tests for chapter 2: unequal splits in the harness.

You already added `split` to split_users; these pin it down and pass it through run_aa and run_ab.
"""

import numpy as np
import pytest

from abkit.harness import multiply, run_aa, run_ab, split_users


@pytest.mark.parametrize("n_users, split, n_treated", [(1000, 0.1, 100), (1000, 0.25, 250), (11, 0.5, 5)])
def test_split_users_treats_the_requested_share(n_users, split, n_treated):
    treated = split_users(n_users, np.random.default_rng(1), split=split)

    assert treated.sum() == n_treated


def recorder(sizes):
    """A stand-in test that records group sizes and returns a fixed p-value."""
    def test(control, treatment):
        sizes.append((len(control), len(treatment)))
        return 0.5
    return test


def test_run_aa_passes_the_split_through():
    sizes = []

    run_aa(np.arange(1000.0), recorder(sizes), n_simulations=5, seed=1, split=0.1)

    assert sizes == [(900, 100)] * 5


def test_run_ab_passes_the_split_through():
    sizes = []

    run_ab(np.arange(1000.0), recorder(sizes), inject=multiply, lift=0.1, n_simulations=5, seed=1, split=0.25)

    assert sizes == [(750, 250)] * 5


def test_the_default_split_is_still_half():
    sizes = []

    run_aa(np.arange(10.0), recorder(sizes), n_simulations=3, seed=1)

    assert sizes == [(5, 5)] * 3
