"""Names of mesh regions: which boundaries are ports, and region patterns.

NGSolve selects regions (``mesh.Materials``, ``mesh.Boundaries``, ``dx``,
``ds``, ``definedon``) with a regular expression, so a material or face name
taken from a CAD file (``Body(1)``, ``window+flange``) must be escaped before
it is used as a pattern.
"""

import re
from typing import Iterable


def is_port_name(name) -> bool:
    """True if a boundary name denotes a port: it starts with ``port``.

    The geometry code numbers ports ``port<N>`` (``port1_substrate`` /
    ``port1_air`` form one composite port); names chosen with
    ``assign_ports()``, such as ``portA``, count too.  ``support`` or
    ``transport`` merely contain "port" and are not ports.
    """
    return bool(name) and str(name).lower().startswith("port")


def region_pattern(names: Iterable[str]) -> str:
    """NGSolve region pattern selecting exactly the given names.

    Each name is escaped, so regular-expression characters in it are matched
    literally: ``region_pattern(['cell(1)', 'a+b'])`` -> ``'cell\\(1\\)|a\\+b'``.
    """
    return "|".join(re.escape(str(n)) for n in names)
