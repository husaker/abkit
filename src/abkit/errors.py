"""Exceptions and warnings that abkit raises or issues on purpose (Effective Python, items 87 and 89).

Every exception abkit raises deliberately inherits from AbkitError, so a caller can catch all of
them with one except clause; bad arguments stay ValueErrors. Every warning inherits from
AbkitWarning, so a caller can filter all of them, or one kind at a time.
"""


class AbkitError(Exception):
    """Base class for every exception that abkit raises on purpose."""


class SampleRatioMismatchError(AbkitError):
    """The group sizes differ from the design by more than chance allows."""


class AbkitWarning(UserWarning):
    """Base class for every warning that abkit issues."""


class SampleRatioMismatchWarning(AbkitWarning):
    """A sample ratio mismatch that the caller asked to be warned about instead of stopped."""


class SmallExpectedCountWarning(AbkitWarning):
    """An expected count below five, where the chi-square p-value is unreliable."""