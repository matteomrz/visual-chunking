import json
import logging
from pathlib import Path
from typing import Any

from pandas import Series
from PIL.Image import Image
from datasets import load_dataset
from faster_coco_eval import COCO, COCOeval_faster
from tqdm import tqdm

from config import CONFIG_DIR, GUIDELINES_DIR
from lib.parsing.model.document_parser import DocumentParser
from lib.parsing.model.options import ParserOptions
from lib.parsing.model.parsing_result import ParsingBoundingBox, ParsingMetaData, ParsingResult, \
    ParsingResultType
from lib.utils.f1_coco import get_f1_metrics
from lib.utils.thesis_names import get_parser_thesis_name
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


def publaynet_gt_exists(expected_cnt: int) -> bool:
    """Check if the Ground truth file already exists for a number of items."""

    if PUBLAYNET_GT_PATH.exists():
        with open(PUBLAYNET_GT_PATH, "r") as gt:
            try:
                gt_obj = json.load(gt)
                # Check if the object is a valid coco file
                COCO(PUBLAYNET_GT_PATH)

                item_cnt = len(gt_obj["images"])
                return item_cnt == expected_cnt

            except BaseException as e:
                logger.warning(
                    f"Malformed PubLayNet ground truth file at: {PUBLAYNET_GT_PATH}. "
                    f"Error: {str(e)}"
                )

    return False


def create_publaynet_gt(max_items: int = 200, exist_ok: bool = False):
    """
    Creates the ground truth file for the COCO Evaluation of PubLayNet.
    The ground truth must follow the format from: https://cocodataset.org/#format-data

    Args:
        max_items: How many rows from the dataset should be included for the evaluation.
        Set to <=0 to include all. Default: 200.
        exist_ok: Skip creating the ground truth file if it already exists.
        Skips if the file has the same amount of items. Default: False
    """

    if exist_ok and publaynet_gt_exists(max_items):
        logger.info(f"Ground truth file already exists at: {PUBLAYNET_GT_PATH}")
        return

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


def _merge_list_items(node: ParsingResult):
    """Merges orphaned list items into overarching list groups."""

    list_start = 0
    list_items = []

    def wrap_in_list():
        list_node = ParsingResult(
            type=ParsingResultType.LIST,
            content="",
            geom=[],
            id="",  # ID does not matter for PubLayNet
            children=list_items,
            parent=node,
        )

        for item in list_items:
            item.parent = list_node

        node.children.insert(list_start, list_node)

    idx = 0
    while idx < len(node.children):
        child = node.children[idx]

        # We found a list item that does not have a list that wraps around it
        if child.type in [
            ParsingResultType.LIST_ITEM, ParsingResultType.REFERENCE_ITEM
        ]:
            if not list_items:
                list_start = idx

            node.children.pop(idx)
            list_items.append(child)

        else:
            # The previous elements were ungrouped list nodes
            # We wrap them into a dummy list for publaynet
            if list_items:
                wrap_in_list()
                list_items = []

                # Account for newly inserted node
                idx += 1

            # Recurse for any children that should not contain unwrapped list items
            if child.type not in [
                ParsingResultType.LIST, ParsingResultType.REFERENCE_LIST
            ]:
                _merge_list_items(child)

            idx += 1

    # Catch last elements are orphaned list items
    if list_items:
        wrap_in_list()


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
            _merge_list_items(result)
            _add_group_bounding_boxes(result)
            coco = get_coco_annotations(result)

            cocos.extend(coco)

    return cocos


def _create_evaluation(res: ParsingResult, parser: DocumentParser[Any]) -> COCOeval_faster:
    """Create the evaluation object for the DocumentParser."""
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


def get_class_metrics(coco_eval: COCOeval_faster, parser: DocumentParser[Any]) -> tuple[
    Series, Series]:
    """
    Get the per-class and macro F1@50 and F1@50:95.

    Args:
        coco_eval: COCOEval_faster returned from evaluate_parser()
        parser: The DocumentParser belonging to the evaluation

    Returns:
        (Series, Series): Tuple of Series (F1@50, F1@50:95) per category and macro.
        Name is set to the Parsers name.
    """

    classes = get_f1_metrics(coco_eval)
    parser_name = get_parser_thesis_name(parser.module)

    f_50 = {
        c["class"]: c["f1@50"]
        for c in classes
    }
    f_50_series = Series(f_50, name=parser_name)

    f_50_95 = {
        c["class"]: c["f1@50:95"]
        for c in classes
    }
    f_50_95_series = Series(f_50_95, name=parser_name)

    return f_50_series, f_50_95_series


def evaluate_parser(parser: DocumentParser[Any]) -> COCOeval_faster:
    """
    Evaluates a DocumentParser on the PubLayNet dataset.

    Returns:
        COCOEval_faster containing the evaluation results
    """

    if not PUBLAYNET_GT_PATH.exists():
        logger.info(f"Missing PubLayNet ground truth file at: {PUBLAYNET_GT_PATH}."
                    "Generating new ground truth file with default settings.")
        create_publaynet_gt()

    # Process PubLayNet files
    logger.info(f"Evaluating {parser.module.value} on PubLayNet...")
    results = parser.process_batch(PUBLAYNET_NAME, _parsing_options)
    coco_eval = _create_evaluation(results[0], parser)

    coco_eval.params.useCats = True

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    logger.info(f"Finished evaluating {parser.module.value}.")
    return coco_eval
