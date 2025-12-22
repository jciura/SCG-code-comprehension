from enum import Enum


class RelationTypes(Enum):
    """Allowed to filter neighbors based of type of relation between them in related_entities.py"""

    DECLARATION = "DECLARATION"
    DECLARATION_BY = "DECLARATION_BY"
    CALL = "CALL"
    CALL_BY = "CALL_BY"
    RETURN_TYPE_ARGUMENT = "RETURN_TYPE_ARGUMENT"
    RETURN_TYPE_ARGUMENT_BY = "RETURN_TYPE_ARGUMENT_BY"
    RETURN_TYPE = "RETURN_TYPE"
    RETURN_TYPE_BY = "RETURN_TYPE_BY"
    EXTEND = "EXTEND"
    EXTEND_BY = "EXTEND_BY"
    ANY = "ANY"
