from enum import Enum


class QueryTopMode(Enum):
    """Query mode: list only or full description"""

    LIST_ONLY = "list_only"
    FULL_DESC = "full_desc"
