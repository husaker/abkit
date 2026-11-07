from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class SupportsPValue(Protocol):
    @property
    def p_value(self) -> float: ...

class TwoSampleTest(Protocol):
    def __call__(self, control: np.ndarray, treatment: np.ndarray, /) -> SupportsPValue | float: ...