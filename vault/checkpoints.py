"""Model-release gates for SafetySAC checkpoint loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from .model_release import get_model_release

ModelT = TypeVar("ModelT")


def load_checkpoint(model_class: type[ModelT], path: str | Path, **kwargs: Any) -> ModelT:
    """Validate checkpoint provenance before delegating to an SB3 loader."""
    verified = get_model_release().verify_checkpoint(path)
    return model_class.load(str(verified), **kwargs)


def load_safety_sac(path: str | Path, **kwargs: Any):
    from safety_sb3 import SafetySAC

    return load_checkpoint(SafetySAC, path, **kwargs)


def load_reach_avoid_safety_sac(path: str | Path, **kwargs: Any):
    from .reach_avoid_sac import ReachAvoidSafetySAC

    return load_checkpoint(ReachAvoidSafetySAC, path, **kwargs)
