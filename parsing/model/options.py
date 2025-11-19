from enum import Enum


class ParserOptions(Enum):
    """Available Options for DocumentParser"""
    ANNOTATE = "--annotate"
    EXIST_OK = "--exist_ok"
