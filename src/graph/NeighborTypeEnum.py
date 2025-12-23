from enum import Enum


class NeighborTypeEnum(Enum):
    """Allowed to filter neighbor type selected for node in specific_nodes.py."""

    CLASS = "CLASS"
    METHOD = "METHOD"
    CONSTRUCTOR = "CONSTRUCTOR"
    INTERFACE = "INTERFACE"
    TYPE_PARAMETER = "TYPE_PARAMETER"
    ENUM = "ENUM"
    ANY = "ANY"
    OBJECT = "OBJECT"
    TRAIT = "TRAIT"
    TYPE = "TYPE"
