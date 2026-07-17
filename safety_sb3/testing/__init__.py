"""Reference environments for validating the library (and your own setup).

Dependency-light, CPU-trainable in minutes, and chosen so that the two problems
are visibly different -- the property a value unit test cannot check.
"""
from .bicycle5d import Bicycle5D, BicycleGoal

__all__ = ["Bicycle5D", "BicycleGoal"]
