"""Structural types for the parts of abkit that accept any test (PEP 544; Fluent Python, chapter 13)."""

from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class SupportsPValue(Protocol):
    """Any object with a read-only p_value attribute holding a float.

    A protocol, so a class satisfies it by having the attribute, not by inheriting from
    it: abkit's ComparisonResult and MannWhitneyResult both do. The harness uses it to pull
    the p-value out of whatever a test returns. Because it is runtime-checkable,
    isinstance(result, SupportsPValue) works, but it only checks that the attribute exists,
    not that it holds a float. A scipy result, whose attribute is called pvalue, is not a
    SupportsPValue.
    """

    @property
    def p_value(self) -> float:
        """The p-value of the test."""
        ...

class TwoSampleTest(Protocol):
    """A callable that compares a control sample with a treatment sample.

    It takes the two arrays positionally and returns either an object with a p_value
    attribute or a bare p-value. The parameters are positional-only, so a conforming
    function may name them as it likes; welch_t_test, z_test_proportions and mann_whitney
    all satisfy it, and run_aa and run_ab accept any of them.
    """

    def __call__(self, control: np.ndarray, treatment: np.ndarray, /) -> SupportsPValue | float:
        """Compare the two samples; return a result with a p_value, or the p-value itself."""
        ...