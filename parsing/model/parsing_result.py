from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field


class ParsingResultType(Enum):
    ROOT = "__root__"
    TITLE = "title"
    PARAGRAPH = "text"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FOOTNOTE = "footer"
    HEADING = "header"
    FIGURE = "image"


@dataclass
class ParsingBoundingBox:
    """
    Bounding Box for an element in the Parsing Result.
    LTRB format with top left coordinate source.
    Values between 0.0 and 1.0. (Fraction of width / height)
    """

    page: int
    left: float
    top: float
    right: float
    bottom: float

    def to_json(self) -> dict:
        return {
            "page": self.page,
            "l": self.left,
            "t": self.top,
            "r": self.right,
            "b": self.bottom,
        }


@dataclass
class ParsingResult:
    """Parsing Result from a PDF parser"""

    id: str
    type: ParsingResultType | str
    content: str
    geom: list[ParsingBoundingBox]
    image: str = None
    children: list[ParsingResult | None] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def root(cls, metadata=None) -> ParsingResult:
        """Creates the root of the Parsing Result"""
        if not metadata:
            metadata = {}

        return cls(
            id="__root__",
            type=ParsingResultType.ROOT,
            content="",
            geom=[],
            metadata=metadata,
        )

    def to_json(self) -> dict[str, str | dict | list]:
        type_name = self.type
        if isinstance(type_name, ParsingResultType):
            type_name = type_name.name

        res: dict[str, str | dict | list] = {
            "id": self.id,
            "type": type_name,
            "content": self.content,
            "geom": [bbox.to_json() for bbox in self.geom],
        }

        if self.metadata:
            res["metadata"] = self.metadata
        if self.image:
            res["image"] = self.image
        if len(self.children) > 0:
            res["children"] = [child.to_json() for child in self.children]

        return res
