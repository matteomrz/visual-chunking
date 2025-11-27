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
    UNKNOWN = "unknown"

    @classmethod
    def get_type(cls, name: str) -> ParsingResultType | str:
        try:
            return cls[name]
        except KeyError:
            if not name:
                return cls.UNKNOWN
            else:
                return name


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

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "l": self.left,
            "t": self.top,
            "r": self.right,
            "b": self.bottom,
        }

    @classmethod
    def from_dict(cls, dictionary: dict) -> ParsingBoundingBox:
        try:
            return cls(
                page=dictionary["page"],
                left=dictionary["l"],
                top=dictionary["t"],
                right=dictionary["r"],
                bottom=dictionary["b"],
            )
        except KeyError as e:
            raise ValueError(f"Error: Missing key in BoundingBox dictionary: {e}")
        except TypeError as e:
            raise ValueError(f"Error: Invalid type in BoundingBox dictionary: {e}")


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

    def to_dict(self) -> dict[str, str | dict | list]:
        type_name = self.type
        if isinstance(type_name, ParsingResultType):
            type_name = type_name.name

        res: dict[str, str | dict | list] = {
            "id": self.id,
            "type": type_name,
            "content": self.content,
            "geom": [bbox.to_dict() for bbox in self.geom],
        }

        if self.metadata:
            res["metadata"] = self.metadata
        if self.image:
            res["image"] = self.image
        if len(self.children) > 0:
            res["children"] = [child.to_dict() for child in self.children]

        return res

    @classmethod
    def from_dict(cls, dictionary: dict) -> ParsingResult:
        try:
            elem_id: str = dictionary["id"]
            content: str = dictionary["content"]
            geom: list[dict] = dictionary["geom"]
            type_name: str = dictionary["type"]

            metadata: dict = dictionary.get("metadata", {})
            children: list[dict] = dictionary.get("children", [])
            image: str | None = dictionary.get("image", None)
        except KeyError as e:
            raise ValueError(f"Error: Missing key in ParsingResult dictionary: {e}")
        except TypeError as e:
            raise ValueError(f"Error: Invalid type in ParsingResult dictionary: {e}")

        parsing_type = ParsingResultType.get_type(type_name)

        geom_parsed: list[ParsingBoundingBox] = [
            ParsingBoundingBox.from_dict(bbox)
            for bbox in geom
        ]

        children_parsed = [
            cls.from_dict(child)
            for child in children
        ]

        return cls(
            id=elem_id,
            type=parsing_type,
            content=content,
            geom=geom_parsed,
            image=image,
            metadata=metadata,
            children=children_parsed
        )
