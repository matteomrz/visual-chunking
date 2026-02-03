import json
from logging import getLogger
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from lib.parsing.methods.parsers import Parsers
from lib.parsing.methods.vlm import VLMParser
from lib.parsing.methods.vlm_prompt import VLM_MD_PROMPT, get_prompt_for_page_wise
from lib.utils.json_trim import trim_json_string
from lib.utils.pdf_to_page_img import pdf_to_page_img_bytes

logger = getLogger(__name__)


class GeminiParser(VLMParser):
    """Uses Google's Gemini Model for Document Parsing."""

    module = Parsers.GEMINI

    client: genai.Client
    model_name: str
    max_retries: int

    def __init__(self, **kwargs):
        load_dotenv()
        self.client = genai.Client()
        self.model_name = kwargs.get("model_name", "gemini-2.5-flash")
        self.max_retries = kwargs.get("max_retries", 2)

    def _parse(self, file_path: Path, options: dict = None) -> dict:
        page_image_bytes = pdf_to_page_img_bytes(file_path, "jpeg")

        results = {"layout_elements": []}

        for idx, page_bytes in enumerate(page_image_bytes):
            doc_part = types.Part.from_bytes(data=page_bytes, mime_type="image/jpeg")
            prompt = get_prompt_for_page_wise(idx + 1)

            for retry in range(self.max_retries):
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        doc_part,
                        prompt
                    ]
                )

                try:
                    res_json = trim_json_string(response.text)

                    res = json.loads(res_json)
                    res_elems = res["layout_elements"]
                    results["layout_elements"].extend(res_elems)

                    break
                except BaseException as e:
                    malformed_msg = f"Malformed response from {self.module.value}. "

                    if retry == self.max_retries - 1:
                        error_msg = f"Error: {e}. Received response: {response.text}"
                        raise ValueError(malformed_msg + error_msg)
                    else:
                        retry_msg = f"Retrying {file_path.stem}... ({retry + 1}/{self.max_retries})"
                        logger.warning(malformed_msg + retry_msg)

        return results

    def _get_md(self, raw_result: dict, file_path: Path) -> str:
        doc_bytes = file_path.read_bytes()
        doc_part = types.Part.from_bytes(
            data=doc_bytes,
            mime_type="application/pdf"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[doc_part, VLM_MD_PROMPT]
        )

        return response.text
