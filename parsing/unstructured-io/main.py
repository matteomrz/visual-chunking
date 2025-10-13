import argparse
import json
import os

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

from config import BOUNDING_BOX_DIR, DEFAULT_GUIDELINE, GUIDELINES_DIR
from parsing.draw import draw_bboxes

MODULE = "unstructured-io"

src_path = GUIDELINES_DIR
dst_path = BOUNDING_BOX_DIR / MODULE

valid_strats = ["auto", "hi_res", "fast", "ocr_only"]
default_strat = "hi_res"

parser = argparse.ArgumentParser()

parser.add_argument("--draw", "-d", action="store_true", help="Create annotated PDF")
parser.add_argument(
    "--file",
    "-f",
    type=str,
    default=DEFAULT_GUIDELINE,
    help=f'PDF filename without extension. Default: "{DEFAULT_GUIDELINE}"',
)
parser.add_argument(
    "--strat",
    "-s",
    type=str,
    default=default_strat,
    choices=valid_strats,
    help=f'Partitioning Strategy. Default: "{default_strat}"',
)

args = parser.parse_args()


def main():
    """Extract partitions from guideline PDF using unstructured.io"""
    strat = args.strat
    base_file_name = args.file
    full_src = src_path / f"{base_file_name}.pdf"

    if not full_src.exists():
        raise FileNotFoundError(f"Error: PDF not found: {full_src}")

    print(f"Parsing {full_src.name} using {MODULE}...")
    elements = partition_pdf(
        filename=full_src,
        strategy=strat,
        languages=["eng", "deu"],
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=dst_path / "images" / base_file_name / strat,
    )

    _transform_and_save(elements_to_json(elements=elements))

    if args.draw:
        print("Drawing bounding boxes...")
        draw_bboxes(base_file_name, module_name=MODULE, appendix=f'-{strat}')


def _transform_and_save(json_string):
    """Transform the output to the schema specified in bounding_boxes/schema.json and saves it as a json file"""
    parsed = json.loads(json_string)
    json_elements = []

    for element in parsed:
        transformed = {
            "content": element["text"],
            "metadata": {
                "element_id": element["element_id"],
                "type": element["type"],
                "layout": {
                    "page_num": element["metadata"]["page_number"],
                    "height": element["metadata"]["coordinates"]["layout_height"],
                    "width": element["metadata"]["coordinates"]["layout_width"],
                    "bbox": element["metadata"]["coordinates"]["points"],
                },
            }
        }
        if "image_path" in element["metadata"]:
            transformed["image_path"] = element["metadata"]["image_path"]

        json_elements.append(transformed)

    os.makedirs(dst_path, exist_ok=True)

    output_path = dst_path / f"{args.file}-{args.strat}-output.json"
    with open(output_path, "w") as f:
        json.dump(json_elements, f, indent=4)
        print(f"Success: JSON saved at: {output_path}")


if __name__ == "__main__":
    main()
