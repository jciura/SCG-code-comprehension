from enum import Enum


class NeighborTypeEnum(Enum):
    """Allowed to filter neighbor type selected for node in specific_nodes.py."""

    CLASS = "CLASS"
    METHOD = "METHOD"
    CONSTRUCTOR = "CONSTRUCTOR"
    VARIABLE = "VARIABLE"
    PARAMETER = "PARAMETER"
    ANY = "ANY"
