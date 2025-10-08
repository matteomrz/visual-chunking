import os
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from draw import draw_bboxes
import argparse

src_path = "../guidelines"
dst_path = "../bounding-boxes/unstructured-io"
valid_strats = ["auto", "hi_res", "fast", "ocr_only"]
default_strat = "hi_res"
default_file = "example-guideline"

parser = argparse.ArgumentParser()

parser.add_argument("--draw", "-d", action="store_true", help="Create annotated PDF")
parser.add_argument(
    "--file",
    "-f",
    type=str,
    default=default_file,
    help=f'PDF filename without extension. Default: "{default_file}"',
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
    full_src = f"{src_path}/{base_file_name}.pdf"
    output_path = f"{dst_path}/{base_file_name}-{strat}-output.json"

    if not os.path.exists(full_src):
        print(f"Error: PDF not found: {full_src}")
        return

    elements = partition_pdf(
        filename=full_src,
        strategy=strat,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=f"{dst_path}/images",
    )

    elements_to_json(elements=elements, filename=output_path)

    print(f"Success: JSON saved at: {output_path}")

    if args.draw:
        draw_bboxes(base_file_name)


if __name__ == "__main__":
    main()
