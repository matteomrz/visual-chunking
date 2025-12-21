import os
from pathlib import Path

from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as document_ai
from google.cloud.documentai_v1 import Document

from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.document_parser import DocumentParser
from lib.parsing.model.parsing_result import ParsingResult

PDF_MIME_TYPE = "application/pdf"


class DocumentAIParser(DocumentParser[Document]):
    """Uses the LayoutParser from Google's Document AI for parsing."""

    module = Parsers.DOCUMENT_AI

    client: document_ai.DocumentProcessorServiceClient
    processor_path: str

    def __init__(self):
        load_dotenv()
        location = os.getenv("DOC_AI_LOCATION")
        project = os.getenv("DOC_AI_LOCATION")
        processor = os.getenv("DOC_AI_LOCATION")

        client_options = ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com"
        )

        self.client = document_ai.DocumentProcessorServiceClient(
            client_options=client_options
        )

        self.processor_path = self.client.processor_path(project, location, processor)

    def _parse(self, file_path: Path, options: dict = None):
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        raw_doc = document_ai.RawDocument(content=file_bytes, mime_type=PDF_MIME_TYPE)
        request = document_ai.ProcessRequest(name=self.processor_path, raw_docuemnt=raw_doc)

        response = self.client.process_document(request)

        return response.document

    def _get_md(self, raw_result: Document, file_path: Path) -> str:
        return raw_result.text
