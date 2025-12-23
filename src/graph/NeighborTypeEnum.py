from enum import Enum


class NeighborTypeEnum(Enum):
    """Allowed to filter neighbor type selected for node in specific_nodes.py."""

    CLASS = "CLASS"
    METHOD = "METHOD"
    CONSTRUCTOR = "CONSTRUCTOR"
    VARIABLE = "VARIABLE"
    PARAMETER = "PARAMETER"
    VALUE = "VALUE"
    INTERFACE = "INTERFACE"
    TYPE_PARAMETER = "TYPE_PARAMETER"
    ENUM = "ENUM"
    ANY = "ANY"
    OBJECT = "OBJECT"
    TRAIT = "TRAIT"
    SELF_PARAMETER = "SELF_PARAMETER"
    TYPE = "TYPE"
