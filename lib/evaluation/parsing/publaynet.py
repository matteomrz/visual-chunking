import json
import logging
from pathlib import Path
from typing import Any

from PIL.Image import Image
from datasets import load_dataset
from faster_coco_eval import COCO, COCOeval_faster
from tqdm import tqdm

from config import CONFIG_DIR, GUIDELINES_DIR
from lib.parsing.model.document_parser import DocumentParser
from lib.parsing.model.options import ParserOptions
from lib.parsing.model.parsing_result import ParsingBoundingBox, ParsingMetaData, ParsingResult, \
    ParsingResultType
from lib.utils.merge_boxes import merge_adjacent_boxes
from lib.utils.open import open_parsing_result
from lib.utils.to_coco import get_coco_annotations

logger = logging.getLogger(__name__)

# The original PubLayNet dataset contains 146,874 sample images.
# There is no longer a publicly available source to obtain the dataset in its entirety.
# For our purposes, we use a subset of the original dataset with 500 samples.
# Dataset source: https://huggingface.co/datasets/kenza-ily/publaynet-mini
PUBLAYNET_DATASET = "kenza-ily/publaynet-mini"
PUBLAYNET_SPLIT = "train"

PUBLAYNET_NAME = "publaynet"
# Samples are saved  in PDF format to this directory
PUBLAYNET_DIR = GUIDELINES_DIR / PUBLAYNET_NAME

# Ground Truth and Prediction Data are saved here
PUBLAYNET_CONFIG_DIR = CONFIG_DIR / PUBLAYNET_NAME
PUBLAYNET_GT_PATH = PUBLAYNET_CONFIG_DIR / "gt.json"

# Mapping from category id to category name
PUBLAYNET_CATEGORIES = {
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"
}


def create_publaynet_gt(max_items: int = 200):
    """
    Creates the ground truth file for the COCO Evaluation of PubLayNet.
    The ground truth must follow the format from: https://cocodataset.org/#format-data

    Args:
        max_items: How many rows from the dataset should be included for the evaluation.
        Set to <=0 to include all. Default: 200.
    """

    logger.info("Started generating new PubLayNet ground truth file.")
    logger.debug(f"Downloading dataset from {PUBLAYNET_DATASET}...")

    dataset = load_dataset(PUBLAYNET_DATASET, split=PUBLAYNET_SPLIT)
    PUBLAYNET_DIR.mkdir(parents=True, exist_ok=True)
    PUBLAYNET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if max_items > 0:
        item_cnt = min(max_items, len(dataset))
    else:
        item_cnt = len(dataset)

    images = []
    annotations = []

    logger.debug(f"Processing PubLayNet dataset...")
    for i in tqdm(range(item_cnt)):
        item = dataset[i]
        item_id = int(item["id"])

        # Save images as PDF files
        img: Image = item["image"]
        pdf_path = PUBLAYNET_DIR / f"{item_id}.pdf"
        img.save(pdf_path)

        image_entry = {
            "id": item_id,
            "file_name": f"{item_id}.jpg",
            "height": img.height,
            "width": img.width
        }
        images.append(image_entry)

        # Remove segmentation to save space
        # We only need bounding boxes for evaluation
        for annotation in item["annotations"]:
            del annotation["segmentation"]
            annotations.append(annotation)

    categories = [
        {"id": k, "name": v}
        for k, v in PUBLAYNET_CATEGORIES.items()
    ]

    gt_obj = {
        "images": images,
        "categories": categories,
        "annotations": annotations
    }

    with open(PUBLAYNET_GT_PATH, "w") as f:
        json.dump(gt_obj, f, indent=2)
        logger.info(f"Saved ground truth file to {PUBLAYNET_GT_PATH}")


def _add_group_bounding_boxes(result: ParsingResult):
    """
    In some cases group objects (like lists) do not have their own bounding box.
    Set geometry to the union of their children's bounding boxes.
    """

    if result.type != ParsingResultType.ROOT and not result.geom:
        geom = []
        for child in result.children:
            geom.extend(child.geom)

        if not geom:
            geom.append(ParsingBoundingBox.origin())
            logger.warning("Found element without geometric information."
                           "Setting geom to page origin...")

        # Column aware merging of the children's bounding boxes
        result.geom = merge_adjacent_boxes(geom)

    for child in result.children:
        _add_group_bounding_boxes(child)


def _create_dt(result_dir: Path) -> list:
    """Create the content of the prediction file for the output in result_dir."""
    if not result_dir.is_dir():
        raise ValueError(f"No directory found at: {result_dir}")

    cocos = []

    with open(PUBLAYNET_GT_PATH, "r") as gt:
        gt_obj = json.load(gt)
        images = [img["file_name"] for img in gt_obj["images"]]

        for image in images:
            img_path = result_dir / image
            output_path = img_path.with_suffix(".json")

            if not output_path.exists():
                raise FileNotFoundError(f"Missing Parsing Output at: {output_path}")

            result = open_parsing_result(output_path)
            _add_group_bounding_boxes(result)
            coco = get_coco_annotations(result)

            cocos.extend(coco)

    return cocos


def _create_evaluation(res: ParsingResult, parser: DocumentParser[Any]) -> COCOeval_faster:
    res_path = res.metadata[ParsingMetaData.JSON_PATH.value]
    res_dir = Path(res_path).parent

    parser_name = parser.module.value
    dt_path = PUBLAYNET_CONFIG_DIR / f"dt_{parser_name}.json"

    dt_obj = _create_dt(res_dir)
    with open(dt_path, "w") as dt:
        json.dump(dt_obj, dt)

    gt = COCO(PUBLAYNET_GT_PATH)
    dt = gt.loadRes(dt_path)

    return COCOeval_faster(gt, dt, "bbox", extra_calc=True, use_area=False)


_parsing_options = {
    ParserOptions.EXIST_OK: True
}


def evaluate_parser(parser: DocumentParser[Any]):
    if not PUBLAYNET_GT_PATH.exists():
        logger.info(f"Missing PubLayNet ground truth file at: {PUBLAYNET_GT_PATH}."
                    "Generating new ground truth file with default settings.")
        create_publaynet_gt()

    # Process PubLayNet files
    results = parser.process_batch(PUBLAYNET_NAME, _parsing_options)
    coco_eval = _create_evaluation(results[0], parser)

    coco_eval.params.useCats = True

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
