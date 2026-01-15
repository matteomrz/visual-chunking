import json
from logging import getLogger
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from lib.parsing.methods.parsers import Parsers
from lib.parsing.methods.vlm import VLMParser
from lib.parsing.methods.vlm_prompt import get_prompt_for_page_wise
from lib.utils.json_trim import trim_json_string
from lib.utils.pdf_to_page_img import pdf_to_page_img_bytes

logger = getLogger(__name__)


class GeminiParser(VLMParser):
    """Uses Google's Gemini Model for Document Parsing."""

    module = Parsers.GEMINI

    client: genai.Client
    model_name: str

    # model_config: types.GenerateContentConfig

    def __init__(self):
        self.client = genai.Client()
        self.model_name = "gemini-2.5-flash-image"  # TODO: Specify date
        # self.model_config = types.GenerateContentConfig(
        #     media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
        # )

    def _parse(self, file_path: Path, options: dict = None) -> dict:
        page_image_bytes = pdf_to_page_img_bytes(file_path, "jpeg")

        results = {"layout_elements": []}

        for idx, page_bytes in enumerate(page_image_bytes):
            doc_part = types.Part.from_bytes(data=page_bytes, mime_type="image/jpeg")
            prompt = get_prompt_for_page_wise(idx + 1)
            response = self.client.models.generate_content(
                model=self.model_name,
                # config=self.model_config,
                contents=[
                    doc_part,
                    prompt
                ]
            )

            res_json = trim_json_string(response.text)

            try:
                res = json.loads(res_json)
                res_elems = res["layout_elements"]
                results["layout_elements"].extend(res_elems)
            except BaseException as e:
                logger.error(
                    f"Malformed response from {self.module.value}. "
                    f"Error: {e}. Received response: {response.text}"
                )

        return results
