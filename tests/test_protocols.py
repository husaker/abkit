"""Chapter 6's engineering checks: one interface for every test in abkit, written as protocols.

``TwoSampleTest`` is checked statically, by running mypy on a small file that assigns each test
function to it; ``SupportsPValue`` is checked at runtime with isinstance, and the harness must accept
any test that returns an object with a ``p_value``.
"""

import importlib.util
import inspect
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from abkit.harness import multiply, run_aa, run_ab
from abkit.mannwhitney import MannWhitneyResult, mann_whitney
from abkit.protocols import SupportsPValue, TwoSampleTest
from abkit.ttest import welch_t_test
from abkit.ztest import ComparisonResult, z_test_proportions

PROJECT = Path(__file__).parents[1]

PREAMBLE = """
    import numpy as np
    from abkit.harness import run_aa
    from abkit.mannwhitney import mann_whitney
    from abkit.protocols import TwoSampleTest
    from abkit.ttest import welch_t_test
    from abkit.ztest import z_test_proportions
"""


def mypy_result(code, tmp_path):
    """Run mypy on a small module written to tmp_path; return (accepted, output)."""
    assert importlib.util.find_spec("mypy") is not None, "mypy isn't installed: run `uv add --dev mypy`"
    check = tmp_path / "protocol_check.py"
    check.write_text(textwrap.dedent(PREAMBLE) + textwrap.dedent(code))
    completed = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", str(PROJECT / "pyproject.toml"), str(check)],
        cwd=PROJECT, capture_output=True, text=True,
        env={**os.environ, "MYPYPATH": str(PROJECT / "src")},  # so mypy reads abkit from src/
    )
    return completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture
def values():
    return np.random.default_rng(6).lognormal(1.0, 1.0, size=400)


# ---------------------------------------------------------------- the static protocol

def test_every_array_test_in_abkit_satisfies_two_sample_test(tmp_path):
    accepted, output = mypy_result("""
        t1: TwoSampleTest = welch_t_test
        t2: TwoSampleTest = z_test_proportions
        t3: TwoSampleTest = mann_whitney
        run_aa(np.zeros(10), welch_t_test, n_simulations=2)
        run_aa(np.zeros(10), mann_whitney, n_simulations=2)
    """, tmp_path)
    assert accepted, output


def test_parameter_names_do_not_matter_because_the_protocol_is_positional_only(tmp_path):
    accepted, output = mypy_result("""
        def named_a_and_b(a: np.ndarray, b: np.ndarray) -> float:
            return 0.5
        t: TwoSampleTest = named_a_and_b
    """, tmp_path)
    assert accepted, output


def test_a_test_with_a_required_third_parameter_is_rejected(tmp_path):
    accepted, output = mypy_result("""
        def needs_alpha(control: np.ndarray, treatment: np.ndarray, alpha: float) -> float:
            return 0.5
        t: TwoSampleTest = needs_alpha
    """, tmp_path)
    assert not accepted
    assert "TwoSampleTest" in output


def test_a_test_returning_something_without_a_p_value_is_rejected(tmp_path):
    accepted, output = mypy_result("""
        def returns_text(control: np.ndarray, treatment: np.ndarray) -> str:
            return "significant"
        t: TwoSampleTest = returns_text
    """, tmp_path)
    assert not accepted
    assert "TwoSampleTest" in output


def test_the_protocol_parameters_are_positional_only():
    parameters = [p for name, p in inspect.signature(TwoSampleTest.__call__).parameters.items() if name != "self"]
    assert [p.name for p in parameters] == ["control", "treatment"]
    assert all(p.kind is inspect.Parameter.POSITIONAL_ONLY for p in parameters)


# ---------------------------------------------------------------- the runtime-checkable protocol

def test_abkit_results_satisfy_supports_p_value_at_runtime(values):
    assert isinstance(welch_t_test(values[:200], values[200:]), SupportsPValue)
    assert isinstance(mann_whitney(values[:200], values[200:]), SupportsPValue)
    assert isinstance(z_test_proportions(np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0])), SupportsPValue)


def test_a_class_satisfies_the_protocol_without_inheriting_from_it():
    @dataclass(frozen=True)
    class MyResult:
        p_value: float

    assert isinstance(MyResult(0.2), SupportsPValue)
    assert SupportsPValue not in MyResult.__mro__  # structural: no inheritance was needed
    # issubclass() is not available for a protocol with a data member such as p_value
    with pytest.raises(TypeError):
        issubclass(MyResult, SupportsPValue)


def test_a_scipy_result_does_not_satisfy_the_protocol_because_its_attribute_is_named_pvalue(values):
    scipy_result = stats.ttest_ind(values[:200], values[200:], equal_var=False)
    assert hasattr(scipy_result, "pvalue")
    assert not isinstance(scipy_result, SupportsPValue)
    assert not isinstance(0.03, SupportsPValue)


# ---------------------------------------------------------------- the harness uses the protocol

def test_the_harness_accepts_a_test_that_returns_a_result(values):
    direct = run_aa(values, welch_t_test, n_simulations=40, seed=3)
    wrapped = run_aa(values, lambda c, t: welch_t_test(c, t).p_value, n_simulations=40, seed=3)
    np.testing.assert_array_equal(direct.p_values, wrapped.p_values)


def test_the_harness_accepts_mann_whitney_directly(values):
    result = run_aa(values, mann_whitney, n_simulations=40, seed=4)
    assert result.n_simulations == 40
    assert np.all((0 <= result.p_values) & (result.p_values <= 1))


def test_run_ab_accepts_a_test_that_returns_a_result(values):
    result = run_ab(values, welch_t_test, inject=multiply, lift=0.0, n_simulations=20, seed=5)
    assert result.n_simulations == 20


def test_the_harness_still_accepts_a_bare_p_value(values):
    as_float = run_aa(values, lambda c, t: float(welch_t_test(c, t).p_value), n_simulations=20, seed=6)
    as_numpy = run_aa(values, lambda c, t: np.float64(welch_t_test(c, t).p_value), n_simulations=20, seed=6)
    np.testing.assert_array_equal(as_float.p_values, as_numpy.p_values)


def test_the_harness_rejects_a_result_without_a_p_value(values):
    def scipy_welch(control, treatment):
        return stats.ttest_ind(control, treatment, equal_var=False)

    with pytest.raises(TypeError, match="p_value"):
        run_aa(values, scipy_welch, n_simulations=2, seed=7)
    with pytest.raises(TypeError, match="p_value"):
        run_aa(values, lambda c, t: None, n_simulations=2, seed=7)
