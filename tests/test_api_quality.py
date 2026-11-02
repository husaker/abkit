"""Chapter 3's engineering checks: every public function and class in abkit is documented and typed.

These tests read your code rather than run it. They go red for every missing docstring or type hint,
listing what's missing, and the last one runs mypy over src/.
"""

import importlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

MODULES = [
    "abkit.harness",
    "abkit.ztest",
    "abkit.ttest",
    "abkit.capping",
    "abkit.regression",
    "abkit.power",
    "abkit.bootstrap",
    "abkit.mannwhitney",
    "abkit.datasets.rees46",
]


def importable(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def public_members(module_name):
    """Functions and classes defined in the module itself, whose names don't start with "_"."""
    if not importable(module_name):
        return  # test_module_exists reports it; the other checks wait until it does
    module = importlib.import_module(module_name)
    for name, obj in vars(module).items():
        defined_here = getattr(obj, "__module__", None) == module_name
        if defined_here and not name.startswith("_") and (inspect.isfunction(obj) or inspect.isclass(obj)):
            yield f"{module_name}.{name}", obj


PUBLIC = [member for module_name in MODULES for member in public_members(module_name)]
FUNCTIONS = [(name, obj) for name, obj in PUBLIC if inspect.isfunction(obj)]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_exists(module_name):
    assert importable(module_name), f"{module_name} can't be imported yet"


@pytest.mark.parametrize("name, obj", PUBLIC, ids=[name for name, _ in PUBLIC])
def test_has_a_docstring(name, obj):
    doc = inspect.getdoc(obj) or ""
    # When a data class has no docstring, @dataclass writes one for you that is just its signature,
    # e.g. "ComparisonResult(estimate: float, ...)". That doesn't count.
    written_by_you = doc and not doc.startswith(obj.__name__ + "(")
    assert written_by_you, f"{name} needs a docstring"
    assert doc.splitlines()[0].rstrip().endswith("."), f"{name}: the summary line should end with a period (PEP 257)"


@pytest.mark.parametrize("name, func", FUNCTIONS, ids=[name for name, _ in FUNCTIONS])
def test_is_fully_annotated(name, func):
    signature = inspect.signature(func)

    missing = [p for p, param in signature.parameters.items() if param.annotation is inspect.Parameter.empty]

    assert not missing, f"{name}: parameters without a type hint: {missing}"
    assert signature.return_annotation is not inspect.Signature.empty, f"{name}: no return type"


def test_mypy_finds_no_errors():
    assert importlib.util.find_spec("mypy") is not None, "mypy isn't installed: run `uv add --dev mypy`"
    project = Path(__file__).parents[1]

    result = subprocess.run([sys.executable, "-m", "mypy", "src"], cwd=project, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout
