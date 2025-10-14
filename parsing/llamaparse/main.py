import argparse
import json
import os

from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_cloud_services.parse import ResultType

from config import BOUNDING_BOX_DIR, DEFAULT_GUIDELINE, GUIDELINES_DIR
from parsing.draw import draw_bboxes

MODULE = "llamaparse"
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

load_dotenv()
key = os.getenv("LLAMAPARSE_API_KEY")

if not key:
    raise ValueError(
        'LLamaParse API Key is not set correctly. Please set "LLAMAPARSE_API_KEY" in the .env file.'
    )

parser = LlamaParse(
    api_key=key,
    result_type=ResultType.JSON,
    extract_layout=True,
    verbose=True,
)


def main():
    full_src = src_path / f'{args.file}.pdf'

    if not full_src.exists():
        raise FileNotFoundError(f"Error: PDF not found: {full_src}")

    print(f"Parsing {full_src.name} using {MODULE}...")
    res = parser.parse(full_src)
    res_json = res.get_json()

    _transform_and_save(res_json)

    if args.draw:
        print("Drawing bounding boxes...")
        draw_bboxes(args.file, MODULE)


def _transform_and_save(json_res):
    """Transform the output to the schema specified in bounding_boxes/schema.json and saves it as a json file"""
    output_path = dst_path / f"{args.file}-output.json"
    json_elements = []
    pages = json_res.get("pages", [])

    for page in pages:
        for element in page["layout"]:
            if not element.get("isLikelyNoise", True):
                type_label = element.get("label", "Unknown")
                height = page.get("height", 0)
                width = page.get("width", 0)

                json_bbox = element.get("bbox", {})
                x = json_bbox.get('x', 0.0) * width
                y = json_bbox.get('y', 0.0) * height
                w = json_bbox.get('w', 0.0) * width
                h = json_bbox.get('h', 0.0) * height

                # Fix the wrongly rotated coordinates -- TODO: Differentiate Angles?
                if page.get("originalOrientationAngle", 0) != 0:
                    x = width - x - w
                    y = height - y - h

                bbox = [
                    [x, y],
                    [x, y + h],
                    [x + w, y + h],
                    [x + w, y]
                ]

                transformed = {
                    "content": element.get("label", "Unknown"),
                    "metadata": {
                        "element_id": element.get("image", "Unknown"),  # doesn't provide any ids... TODO: FIX
                        "type": type_label,
                        "layout": {
                            "page_num": page.get("page", 0),
                            "height": height,
                            "width": width,
                            "bbox": bbox,
                        },
                    }
                }

                json_elements.append(transformed)

    os.makedirs(dst_path, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_elements, f, indent=2)
        print(f"Success: JSON saved at: {output_path}")


if __name__ == "__main__":
    main()
