from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field


class ParsingResultType(Enum):
    ROOT = 0
    TITLE = 1
    PARAGRAPH = 2
    LIST_ITEM = 3
    TABLE = 4
    FOOTNOTE = 5
    HEADING = 6
    FIGURE = 7


@dataclass
class ParsingBoundingBox:
    """Bounding Box for an element in the Parsing Result"""

    page: int
    x: float
    y: float
    width: float
    height: float

    def to_json(self) -> dict:
        return {
            "page": self.page,
            "x": self.x,
            "y": self.y,
            "w": self.width,
            "h": self.height,
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

        if len(self.children) > 0:
            res["children"] = [child.to_json() for child in self.children]
        if self.image:
            res["image"] = self.image
        if self.metadata:
            res["metadata"] = self.metadata

        return res
