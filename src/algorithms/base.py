"""Common interfaces for training algorithms."""

from typing import Protocol


class TrainingAlgorithm(Protocol):
    """Callable interface implemented by algorithm modules."""

    def __call__(self, **kwargs):
        """Run training and return the final state plus output directory."""
        ...
