from dataclasses import dataclass, field

from parsing.model.parsing_result import ParsingBoundingBox


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    geom: list[ParsingBoundingBox] = field(default_factory=list)

    def to_json(self) -> dict[str, str | dict | list]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "geom": [b.to_json() for b in self.geom]
        }


@dataclass
class ChunkingResult:
    chunks: list[Chunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict[str, str | dict | list]:
        return {
            "metadata": self.metadata,
            "chunks": [b.to_json() for b in self.chunks]
        }
