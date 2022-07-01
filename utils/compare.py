"""
    Utilities to compare numbers with precision
"""
from math import isclose


def lt(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    True if a < b given relative and absolute precision
    """
    if isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return False
    return a < b


def lte(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
        True if a <= b given relative and absolute precision
    """
    if isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    return a <= b


def gt(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
        True if a > b given relative and absolute precision
    """
    if isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return False
    return a > b


def gte(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
        True if a >= b given relative and absolute precision
    """

    if isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    return a > b


def eq(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
        True if a = b given relative and absolute precision
    """
    return isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
