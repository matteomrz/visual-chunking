from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Generator


class ParsingResultType(Enum):
    ROOT = "__root__"  # The top-level node containing the entire document structure.

    # TEXTS
    TITLE = "title"  # The specific main title of the document.
    PARAGRAPH = "text"  # Standard body text content.
    SECTION_HEADER = "section_header"  # Section headings or subheaders within the text body.
    FOOTNOTE = "footnote"  # Explanatory notes usually placed at the bottom of a page/text.

    # LISTS
    LIST = "list"  # A container node for a list of items.
    LIST_ITEM = "list_item"  # An individual item within a list.
    REFERENCE_LIST = "ref_list"  # A container node for a list of reference items.
    REFERENCE_ITEM = "ref_item"  # An individual item within a reference list.

    # FIGURES AND TABLES
    CAPTION = "caption"  # Descriptive text immediately accompanying a table or figure.
    FIGURE = "image"  # Graphical elements, diagrams, or pictures.
    TABLE = "table"  # A container node for tabular data.
    DOC_INDEX = "doc_index"  # A tabular node containing the TOC or other document information
    TABLE_ROW = "table_row"  # A horizontal row within a table.
    TABLE_CELL = "table_cell"  # An individual cell containing data within a table row.

    # MISCELLANEOUS
    PAGE_FOOTER = "page_footer"  # Repeating page footer (page numbers, copyright, etc.).
    KEY_VALUE = "key_value"  # A specific key-value pair.
    PAGE_HEADER = "page_header"  # Repeating header found at the top of pages (e.g., journal name).
    KEY_VALUE_AREA = "key_value_area"  # A distinct region grouped by key-value pairs (e.g., article info).
    FORM_AREA = "form_area"  # A region indicating form content (e.g., text-fields).
    FORMULA = "formula"  # A mathematical formula
    WATERMARK = "watermark"  # A watermark from the publishing organization

    # FALLBACK
    UNKNOWN = "unknown"  # Used when the parser cannot determine the element type.
    MISSING = "missing"  # Used when label mappings are absent [Debug].

    @classmethod
    def get_type(cls, name: str) -> ParsingResultType:
        try:
            return cls(name)
        except ValueError:
            return cls.UNKNOWN


class ParsingMetaData(Enum):
    GUIDELINE_PATH = "file_path"
    JSON_PATH = "json_path"
    PARSER = "parsing_method"

    # TIME
    PARSING_TIME = "parsing_time"
    TRANSFORMATION_TIME = "transformation_time"

    # ELEMENT INFO
    HEADER_LEVEL = "header_level"


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
    spans: list[ParsingBoundingBox] = field(default_factory=list)

    @classmethod
    def origin(cls) -> ParsingBoundingBox:
        return cls(
            page=1,
            left=0,
            top=0,
            right=0,
            bottom=0
        )

    @classmethod
    def from_dict(cls, dictionary: dict) -> ParsingBoundingBox:
        """Deserialize from dict."""
        try:
            geom_parsed: list[ParsingBoundingBox] = [
                ParsingBoundingBox.from_dict(bbox)
                for bbox in dictionary.get("spans", [])
            ]

            return cls(
                page=dictionary["page"],
                left=dictionary["l"],
                top=dictionary["t"],
                right=dictionary["r"],
                bottom=dictionary["b"],
                spans=geom_parsed,
            )
        except KeyError as e:
            raise ValueError(f"Error: Missing key in BoundingBox dictionary: {e}")
        except TypeError as e:
            raise ValueError(f"Error: Invalid type in BoundingBox dictionary: {e}")

    def to_dict(self) -> dict:
        """Serialize to dict."""
        res: dict[str, float | int | list] = {
            "page": self.page,
            "l": self.left,
            "t": self.top,
            "r": self.right,
            "b": self.bottom,
        }

        if self.spans:
            res["spans"] = [b.to_dict() for b in self.spans]

        return res


@dataclass
class ParsingResult:
    """
    Parsing Result from a PDF parser.
    The document is portrait as a tree structure, with a ROOT node at the top.
    The ROOT node does not contain any visual content of the document.
    """

    id: str
    type: ParsingResultType
    content: str
    geom: list[ParsingBoundingBox]
    parent: ParsingResult | None
    children: list[ParsingResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    image: str = None

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
            parent=None,
            metadata=metadata,
        )

    @classmethod
    def from_dict(cls, dictionary: dict, parent: ParsingResult = None) -> ParsingResult:
        """Deserialize from dict."""
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
            ParsingBoundingBox.from_dict(bbox) for bbox in geom
        ]

        parsed = cls(
            id=elem_id,
            type=parsing_type,
            content=content,
            geom=geom_parsed,
            parent=parent,
            metadata=metadata,
            image=image,
        )

        parsed.children = [cls.from_dict(child, parsed) for child in children]
        return parsed

    def to_dict(self) -> dict[str, str | dict | list]:
        """Serialize to dict."""
        type_name = self.type
        if isinstance(type_name, ParsingResultType):
            type_name = type_name.value

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

    def flatten(self) -> Generator[ParsingResult]:
        """
        Flattens the document tree structure to iterate over the nodes.
        Does not include the Result it is called on.
        """
        for child in self.children:
            yield child
            yield from child.flatten()

    @property
    def geom_count(self) -> int:
        return len(self.geom)

    @property
    def tokens_per_geom(self) -> float:
        # Assumes each bounding box contains the same amount of tokens
        # TODO: Change to depend on the height of the bounding box
        token_cnt = self.metadata.get("token_cnt", 0)
        return token_cnt / self.geom_count

    def __str__(self):
        return self._rstr().strip()

    def _rstr(self):
        """Recursively creates a string of the entire subtree."""
        content = self.content + self.get_delimiter()

        # Table Row already contains all the contents of its children
        if self.type != ParsingResultType.TABLE_ROW:
            for child in self.children:
                content += child._rstr()

        return content

    def add_delimiters(self):
        """
        Adds delimiters to the content of the ParsingResult.
        Used as a preprocessing step for Chunking.
        """
        self.content += self.get_delimiter()
        for child in self.children:
            child.add_delimiters()

    def get_delimiter(self) -> str:
        """
        Gets the appropriate delimiter for the ParsingResult.
        Parents: "\n"
        TableCells: " | "
        Default: "\n\n"
        """
        delimiter = "\n\n"

        if self.content == "":
            delimiter = ""
        elif self.type == ParsingResultType.TABLE_CELL:
            row = self.parent
            if self in row.children[:-1]:
                delimiter = " | "
        elif self.children:
            delimiter = "\n"

        if self.content.endswith(delimiter):
            delimiter = ""

        return delimiter
