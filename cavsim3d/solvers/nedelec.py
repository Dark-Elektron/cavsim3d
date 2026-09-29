"""Nedelec element kind of the H(curl) spaces.

``'first'`` (default)
    ``HCurl(order=p, type1=True)``, the first-kind Nedelec space.  It has the
    same curls; only curl-free (gradient) functions of the highest degree are
    left out, so ``curl E`` is approximated as with ``'second'`` and the
    curl-free part of ``E`` one order lower.  About a third fewer unknowns at
    order 2.  Models saved without the setting were computed with ``'second'``.
``'second'``
    NGSolve's ``HCurl(order=p)``: every vector polynomial of degree ``p``.

Every H(curl) space of one model must be of the same kind: port modes, system
matrices, snapshots and reconstructed fields share its degrees of freedom.
"""

KINDS = ("second", "first")


def check_kind(kind: str) -> str:
    """Validate a ``nedelec`` setting."""
    if kind not in KINDS:
        raise ValueError(f"nedelec must be one of {KINDS}, got {kind!r}")
    return kind


def hcurl_flags(kind: str) -> dict:
    """Keyword arguments for ``ngsolve.HCurl`` that select the element kind."""
    return {"type1": True} if check_kind(kind) == "first" else {}


def kind_of(fes) -> str:
    """Element kind of an existing H(curl) space (also after unpickling)."""
    try:
        return "first" if fes.flags.ToDict().get("type1") else "second"
    except AttributeError:
        return "second"
