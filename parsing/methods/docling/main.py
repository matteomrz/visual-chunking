import argparse
import json
import os

import datasets
from docling.document_converter import DocumentConverter
from config import BOUNDING_BOX_DIR, DEFAULT_GUIDELINE, GUIDELINES_DIR
from parsing.draw import draw_bboxes

MODULE = "docling"
src_path = GUIDELINES_DIR
dst_path = BOUNDING_BOX_DIR / MODULE

parser = argparse.ArgumentParser()

parser.add_argument("--draw", "-d", action="store_true", help="Create annotated PDF")
parser.add_argument(
    "--file",
    "-f",
    type=str,
    default=DEFAULT_GUIDELINE,
    help=f'PDF filename without extension. Default: "{DEFAULT_GUIDELINE}"',
)

args = parser.parse_args()

converter = DocumentConverter()


def main():
    full_src = src_path / f"{args.file}.pdf"

    if not full_src.exists():
        raise FileNotFoundError(f"Error: PDF not found: {full_src}")

    print(f"Parsing {full_src.name} using {MODULE}...")

    result = converter.convert(full_src)
    json_res = result.document.export_to_dict()
    _transform_and_save(json_res)

    if args.draw:
        print("Drawing bounding boxes...")
        draw_bboxes(args.file, MODULE)


def _transform_and_save(json_res):
    """Transform the output to the schema specified in bounding_boxes/schema.json and saves it as a json file"""
    output_path = dst_path / f"{args.file}-output.json"

    os.makedirs(dst_path, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_res, f, indent=2)
        print(f"Success: JSON saved at: {output_path}")


if __name__ == "__main__":
    main()

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from langchain_qdrant import QdrantVectorStore

# Docling imports
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from transformers import AutoTokenizer

from qdrant_client import QdrantClient

import re


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer to work with LangChain"""

    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(documents)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode([query])[0]
        return embedding.tolist()


class DoclingRAGPipeline:
    """RAG Pipeline using Docling for PDF processing"""

    def __init__(
        self, embed_model_id: str = "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT"
    ):
        self.embed_model_id = embed_model_id
        self.MAX_TOKENS = 128
        self.CHAR_OVERLAP = 50  # Approximation for ~24 tokens
        self.setup_converter()
        self.setup_chunker()
        self.chunk_colors = [
            (1.0, 0.42, 0.42),
            (0.31, 0.80, 0.77),
            (0.27, 0.72, 0.82),
            (0.59, 0.81, 0.71),
            (1.0, 0.92, 0.65),
            (0.87, 0.63, 0.87),
            (0.60, 0.85, 0.78),
            (0.97, 0.86, 0.44),
            (0.73, 0.56, 0.81),
            (0.52, 0.76, 0.91),
        ]

    def setup_converter(self):
        """Set up Docling converter without VLM to avoid hallucinations"""
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
            TableFormerMode,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
        )

        # Use standard PDF pipeline instead of VLM
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            generate_page_images=True,  # Don't need images for standard processing
            images_scale=1.0,
            force_backend_text=False,
            do_ocr=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,  # Match cells accurately
                mode=TableFormerMode.ACCURATE,  # Use accurate mode (slower but better)
            ),
        )

        pdf_options = PdfFormatOption(pipeline_options=pipeline_options)

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: pdf_options}
        )

    def setup_chunker(self):
        """Set up hybrid chunker with markdown table serialization"""
        from docling_core.transforms.chunker.hierarchical_chunker import (
            ChunkingDocSerializer,
            ChunkingSerializerProvider,
        )
        from docling_core.transforms.serializer.markdown import (
            MarkdownTableSerializer,
            MarkdownParams,
        )

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.embed_model_id)
        )

        # Custom serialization provider for markdown tables
        class MarkdownSerializerProvider(ChunkingSerializerProvider):
            def get_serializer(self, doc):
                return ChunkingDocSerializer(
                    doc=doc,
                    table_serializer=MarkdownTableSerializer(),
                    params=MarkdownParams(
                        image_placeholder="",
                        strict_text=True,
                        mark_annotations=True,
                    ),
                )

        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=self.MAX_TOKENS,
            merge_peers=True,
            serializer_provider=MarkdownSerializerProvider(),
        )
        self.tokenizer = tokenizer

    def extract_chunk_metadata(
        self,
        chunk,
        chunk_idx: int,
        pdf_path: str,
        category: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract metadata from chunk including guideline and patient specific metadata"""
        metadata = {
            "source": pdf_path,
            "chunk_id": f"chunk_{chunk_idx}",
        }

        # Add category/subfolder metadata if provided
        if category:
            metadata["category"] = category

        # Add extra metadata from caller (guideline_type, cancer_type, patient_id)
        if extra_metadata:
            metadata.update(extra_metadata)

        # Extract bounding box information
        try:
            if (
                hasattr(chunk, "meta")
                and hasattr(chunk.meta, "doc_items")
                and chunk.meta.doc_items
            ):
                all_bboxes = []
                seen_bboxes = set()
                for doc_item in chunk.meta.doc_items:
                    if hasattr(doc_item, "prov") and doc_item.prov:
                        for prov in doc_item.prov:
                            if hasattr(prov, "bbox") and hasattr(prov, "page_no"):
                                bbox = prov.bbox
                                bbox_key = (
                                    prov.page_no,
                                    float(bbox.l),
                                    float(bbox.t),
                                    float(bbox.r),
                                    float(bbox.b),
                                )

                                # Only add if not seen before
                                if bbox_key not in seen_bboxes:
                                    seen_bboxes.add(bbox_key)
                                    all_bboxes.append(
                                        {
                                            "page": prov.page_no,
                                            "bbox_x0": float(bbox.l),
                                            "bbox_y0": float(bbox.t),
                                            "bbox_x1": float(bbox.r),
                                            "bbox_y1": float(bbox.b),
                                            "coord_origin": bbox.coord_origin,
                                        }
                                    )

                # Store all bboxes in metadata
                if all_bboxes:
                    metadata["all_bboxes"] = json.dumps(all_bboxes)
                    # For backward compatibility, also store the first bbox in the old format
                    first_bbox = all_bboxes[0]
                    metadata["page"] = first_bbox["page"]
                    metadata["bbox_x0"] = first_bbox["bbox_x0"]
                    metadata["bbox_y0"] = first_bbox["bbox_y0"]
                    metadata["bbox_x1"] = first_bbox["bbox_x1"]
                    metadata["bbox_y1"] = first_bbox["bbox_y1"]
                    metadata["coord_origin"] = first_bbox["coord_origin"]

            # Extract headings if available
            if (
                hasattr(chunk, "meta")
                and hasattr(chunk.meta, "headings")
                and chunk.meta.headings[0] is not None
            ):
                metadata["headings"] = chunk.meta.headings[0]
                print(f"  ✓ Extracted headings: {metadata['headings']}")
        except Exception as e:
            print(f"Warning: Could not extract bounding boxes from chunk: {e}")
        return metadata

    def _get_organic_overlap(
        self, text: str, max_sentences: int = 2, max_chars: int = 100
    ) -> str:
        """
        Extract the last 1-2 complete sentences from text for overlap

        Args:
            text: The text to extract overlap from
            max_sentences: Maximum number of sentences to include
            max_chars: Maximum character length for overlap

        Returns:
            The overlap text (last 1-2 sentences)
        """

        # Find sentence boundaries (., !, ?, or newline followed by space/newline)
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())

        # Remove empty strings
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        # Take last N sentences
        overlap_sentences = sentences[-max_sentences:]
        overlap = " ".join(overlap_sentences)

        # Recursive reduction if too long
        if len(overlap) > max_chars:
            if max_sentences > 1:
                # Try with fewer sentences
                return self._get_organic_overlap(text, max_sentences - 1, max_chars)
            else:
                # Last resort: take half of the last sentence
                half_length = len(overlap) // 2

                # Try to break at comma first
                break_point = overlap.rfind(",", 0, half_length)

                # If no comma, try space
                if break_point <= 0:
                    break_point = overlap.rfind(" ", 0, half_length)

                # If found a good break point, use it
                if break_point > 0:
                    return overlap[break_point:].strip()
                else:
                    # No good break point, just truncate to max_chars
                    return overlap[-max_chars:]

        return overlap

    def process_pdf(
        self,
        pdf_path: str,
        category: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Process PDF and return LangChain documents with extended metadata"""
        print(f"Processing PDF: {pdf_path}")

        try:
            # Convert document
            result = self.converter.convert(pdf_path)
            doc = result.document
            print(f"  ✓ Document converted successfully")

            # Chunk document
            chunks = list(self.chunker.chunk(doc))
            print(f"  ✓ Created {len(chunks)} chunks")

            # Convert to LangChain documents with metadata
            documents = []
            previous_chunk_text = None

            for idx, chunk in enumerate(chunks):
                # Get chunk text
                chunk_text = self.chunker.contextualize(chunk)

                # Add organic overlap from previous chunk
                if previous_chunk_text:
                    overlap = self._get_organic_overlap(previous_chunk_text)
                    if overlap:
                        chunk_text = overlap + " " + chunk_text

                # Store current chunk for next iteration
                previous_chunk_text = self.chunker.contextualize(chunk)

                # Extract metadata with extended fields
                metadata = self.extract_chunk_metadata(
                    chunk, idx, pdf_path, category, extra_metadata
                )

                # Stop chunking if headings point to references (end of document)
                if not (
                    "headings" in metadata
                    and isinstance(metadata["headings"], str)
                    # extend this with regex to include References/references Literaturverzeichnis, https://doi.org/ etc. and ignore case
                    and re.search(
                        r"References|Literaturverzeichnis|https://doi\.org/",
                        metadata["headings"],
                        re.IGNORECASE,
                    )
                ):
                    documents.append(
                        Document(page_content=chunk_text, metadata=metadata)
                    )

            return documents

        except Exception as e:
            print(f"  ✗ Error processing {pdf_path}: {e}")
            return []

    def process_directory(
        self,
        directory_path: str,
    ) -> List[Document]:
        """
        Process all PDFs with enhanced metadata extraction for Guidelines/Patients

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            List of all documents from all PDFs
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        pdf_files = list(directory.rglob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return []

        print(f"\nFound {len(pdf_files)} PDF files in {directory_path}")
        print("=" * 60)

        all_documents = []
        successful = 0
        failed = 0

        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{idx}/{len(pdf_files)}] {pdf_file.name}")

            extra_metadata = {}
            # Determine folder structure and extract specific metadata
            try:
                relative_path = pdf_file.relative_to(directory)
                parts = relative_path.parts
                category = None
                # Check and set metadata based on folder hierarchy
                if len(parts) > 1:
                    top_folder = parts[0].lower()
                    if top_folder == "guidelines":
                        cancer_type = parts[1]
                        extra_metadata["guideline_type"] = "curated"
                        extra_metadata["cancer_type"] = cancer_type
                        category = cancer_type
                    elif top_folder == "patients":
                        patient_id = parts[1]
                        extra_metadata["patient_id"] = patient_id
                        category = "patient_documents"
                    else:
                        # Default case: category from first level
                        extra_metadata["category"] = parts[0]
            except ValueError:
                pass

            try:
                documents = self.process_pdf(
                    str(pdf_file),
                    category=category,
                    extra_metadata=extra_metadata,
                )
                if documents:
                    all_documents.extend(documents)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                failed += 1

        print("\n" + "=" * 60)
        print(f"Processing complete:")
        print(f"  ✓ Successfully processed: {successful} PDFs")
        print(f"  ✗ Failed: {failed} PDFs")
        print(f"  Total chunks created: {len(all_documents)}")

        # Print category statistics
        categories = {}
        for doc in all_documents:
            cat = doc.metadata.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\nChunks by category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} chunks")

        return all_documents

    def create_vector_db(
        self,
        documents: List[Document],
        collection_name: str = "rag_documents",
        embedding_dim: int = 768,
        overwrite: bool = False,
    ) -> QdrantVectorStore:
        embedding_function = SentenceTransformerEmbeddings(
            model_name=self.embed_model_id
        )

        # Remote Qdrant configuration

        # Create client with timeout
        client = QdrantClient(url=url, api_key=api_key, port=80)

        # delete existing collection if exists
        if client.collection_exists(collection_name):
            if overwrite == True:
                client.delete_collection(collection_name=collection_name)

        # Create collection if necessary
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim, distance=Distance.COSINE
                ),
            )

        # Use the recommended constructor now
        qdrant_vectorstore = QdrantVectorStore.from_documents(
            documents,
            embedding_function,
            collection_name=collection_name,
            url=url,
            api_key=api_key,
            port=80,
        )

        return qdrant_vectorstore

    def highlight_chunks(
        self,
        pdf_path: str,
        chunks: List[Document],
        chunk_indices: int | List[int],
        output_path: Optional[str] = None,
    ) -> str:
        """Highlight specific chunks in PDF"""
        if isinstance(chunk_indices, int):
            chunk_indices = [chunk_indices]

        if output_path is None:
            indices_str = "-".join(str(idx) for idx in chunk_indices)
            output_path = pdf_path.replace(".pdf", f"_highlight_{indices_str}.pdf")

        # Open PDF
        doc = fitz.open(pdf_path)

        for idx in chunk_indices:
            if idx >= len(chunks):
                continue

            chunk = chunks[idx]
            metadata = chunk.metadata

            # Get all bboxes for this chunk (deserialize from JSON)
            all_bboxes = []
            if "all_bboxes" in metadata:
                try:
                    all_bboxes = json.loads(metadata["all_bboxes"])
                except (json.JSONDecodeError, TypeError):
                    all_bboxes = []

            # Fallback to single bbox if all_bboxes not available
            if not all_bboxes and "bbox_x0" in metadata:
                all_bboxes = [
                    {
                        "page": metadata.get("page", 1),
                        "bbox_x0": metadata["bbox_x0"],
                        "bbox_y0": metadata["bbox_y0"],
                        "bbox_x1": metadata["bbox_x1"],
                        "bbox_y1": metadata["bbox_y1"],
                        "coord_origin": metadata.get("coord_origin", "TOPLEFT"),
                    }
                ]

            if not all_bboxes:
                print(f"Warning: No bbox for chunk {idx}, skipping")
                continue

            # Color for this chunk
            color = self.chunk_colors[idx % len(self.chunk_colors)]

            # Highlight all bboxes for this chunk
            for bbox_info in all_bboxes:
                page_no = bbox_info.get("page", 1) - 1
                if page_no < 0 or page_no >= len(doc):
                    continue

                page = doc[page_no]

                # Get coordinates
                x0 = bbox_info["bbox_x0"]
                y0 = bbox_info["bbox_y0"]
                x1 = bbox_info["bbox_x1"]
                y1 = bbox_info["bbox_y1"]

                # Validate rectangle
                if None in (x0, y0, x1, y1) or x0 == x1 or y0 == y1:
                    continue

                coord_origin = bbox_info.get("coord_origin", "TOPLEFT")

                if coord_origin == "BOTTOMLEFT":
                    # Convert from BOTTOMLEFT to TOPLEFT coordinate system
                    page_height = page.rect.height
                    rect = fitz.Rect(
                        x0,
                        page_height - y0,
                        x1,
                        page_height - y1,
                    )
                else:
                    # Already in TOPLEFT system
                    rect = fitz.Rect(x0, y0, x1, y1)

                if rect.is_empty or rect.is_infinite:
                    continue

                # Add highlight
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=color, fill=None)
                annot.set_border(width=2.0)
                annot.set_opacity(0.8)
                annot.set_info(
                    title=f"Chunk {idx}",
                    content=chunk.page_content[:100] + "...",
                )
                annot.update()

        # Save
        doc.save(output_path)
        doc.close()
        print(f"Highlighted PDF saved to: {output_path}")
        return output_path


def create_rag_system(
    pdf_path: Optional[str] = None,
    directory_path: Optional[str] = None,
    model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT",
    collection_name: str = "rag_documents",
) -> Dict[str, Any]:
    """
    Create complete RAG system from PDF(s)
    """
    if not pdf_path and not directory_path:
        raise ValueError("Must provide either pdf_path or directory_path")

    if pdf_path and directory_path:
        raise ValueError("Provide either pdf_path OR directory_path, not both")

    pipeline = DoclingRAGPipeline(embed_model_id=model_name)

    # Process PDF(s)
    if pdf_path:
        print(f"Processing single PDF: {pdf_path}")
        documents = pipeline.process_pdf(pdf_path)
    else:
        print(f"Processing directory: {directory_path}")
        documents = pipeline.process_directory(directory_path)

    if not documents:
        raise ValueError("No documents were successfully processed")

    # Create vector database
    vectordb = pipeline.create_vector_db(
        documents=documents,
        collection_name=collection_name,
        overwrite=False,
    )

    return {
        "pipeline": pipeline,
        "vectordb": vectordb,
        "documents": documents,
    }
