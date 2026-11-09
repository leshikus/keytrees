r"""
The real identification keys discussed in the 1999 papers

Characters are translated from the Russian drafts, which took them from *Flora Sibiri*,
vol. 6 (1993). ``+`` is ``1``, ``-`` is ``0`` and a blank is ``None``.

EXAMPLES::

    >>> from keytrees.core import optimal_cost
    >>> optimal_cost(SAGINA)
    2.0
    >>> sorted(STELLARIA_FREQUENCY.values())[0]
    'disappearing'
"""
from __future__ import annotations

from .core import Node, Table

SAGINA = Table.from_rows(
    ["S. intermedia", "S. nodosa", "S. procumbens", "S. saginoides"],
    {
        "sepals, petals and capsule valves: 4 each (+) or 5 each (-)": [1, 0, 1, 0],
        "petals at most as long as sepals (+) or twice as long (-)": [1, 0, 1, 1],
        "stems rooting (+) or not rooting (-)": [0, 0, 1, 0],
    },
)

DIANTHUS_MIDDLE_SIBERIA = Table.from_rows(
    ["D. deltoides", "D. ramosissimus", "D. repens", "D. superbus s. str.", "D. superbus subsp. sajanensis", "D. versicolor"],
    {
        "teeth on the petal margin are acute": [1, 0, 0, 0, 0, 1],
        "calyx width under 4 mm": [1, 1, 0, 0, 0, 0],
        "calyx length to width ratio under 4": [0, 0, 1, 0, 0, 1],
        "calyx cylindrical": [1, 0, 0, 0, 1, 0],
    },
)

MOEHRINGIA = Table.from_rows(
    ["M. trinervia", "M. umbrosa", "M. lateriflora"],
    {
        "petals clearly shorter than the calyx; lower leaves on long petioles": [1, 0, 0],
        "sepals acute; peduncles one-flowered": [1, 1, 0],
    },
)

STELLARIA_NOVOSIBIRSK = Table.from_rows(
    ["S. hebecalyx", "S. holostea", "S. bungeana", "S. crassifolia", "S. graminea", "S. longifolia", "S. media", "S. palustris"],
    {
        "capsule much longer than the calyx": [0, 0, 0, 1, 1, 1, 1, 0],
        "lower leaves clearly petiolate": [0, 0, 1, 0, 0, 0, 1, 0],
        "seeds wrinkled": [1, None, 0, 1, 1, 0, 0, 1],
        "bracts membranous, with a green vein": [0, 0, 0, 0, 0, 0, 0, 1],
        "sepals with prominent veins": [1, 0, None, 0, 1, 0, None, 0],
    },
)

STELLARIA_FREQUENCY = {
    "S. hebecalyx": "disappearing",
    "S. holostea": "rare",
    "S. bungeana": "not rare",
    "S. crassifolia": "not rare",
    "S. graminea": "not rare",
    "S. longifolia": "not rare",
    "S. media": "not rare",
    "S. palustris": "not rare",
}

#: Frequency order: p_hebecalyx <= p_holostea <= each "not rare" species.
STELLARIA_ORDER = [(0, 1)] + [(1, j) for j in range(2, 8)]


def _leaf(name: str) -> Node:
    return Node(name=name)


def _fork(test: int, yes: Node, no: Node) -> Node:
    return Node(test=str(test), yes=yes, no=no)


#: Key restricted from Flora Sibiri, with depths 2,2,2,3,4,5,6,6.
STELLARIA_FLORA_TREE = _fork(
    1,
    _fork(
        3,
        _leaf("S. holostea"),
        _fork(
            4,
            _fork(5, _fork(6, _fork(7, _leaf("S. hebecalyx"), _leaf("S. graminea")), _leaf("S. palustris")), _leaf("S. longifolia")),
            _leaf("S. crassifolia"),
        ),
    ),
    _fork(2, _leaf("S. media"), _leaf("S. bungeana")),
)

#: Key from the minimax construction in the 1999 paper.
STELLARIA_MINIMAX_TREE = _fork(
    1,
    _fork(3, _fork(5, _leaf("S. graminea"), _leaf("S. crassifolia")), _fork(2, _leaf("S. media"), _leaf("S. longifolia"))),
    _fork(2, _leaf("S. bungeana"), _fork(4, _leaf("S. palustris"), _fork(5, _leaf("S. hebecalyx"), _leaf("S. holostea")))),
)
