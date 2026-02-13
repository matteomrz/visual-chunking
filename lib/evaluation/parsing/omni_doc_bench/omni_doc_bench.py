import json
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
import pymupdf
import pandas as pd
import numpy as np

from config import (
    CONFIG_DIR,
    EVAL_DIR,
    GUIDELINES_DIR,
    MD_DIR,
    OMNI_DOC_PROJECT_PATH
)
from lib.parsing.methods.parsers import Parsers
from lib.utils.thesis_names import get_parser_thesis_name

omni_doc_repo_id = "opendatalab/OmniDocBench"
omni_doc_name = "omni_doc_bench"

omni_doc_dir = GUIDELINES_DIR / omni_doc_name
image_dir = omni_doc_dir / "images"
pdf_dir = omni_doc_dir / "pdfs"
gt_path = omni_doc_dir / "OmniDocBench.json"

schema_path = EVAL_DIR / "parsing" / omni_doc_name / "omni_doc_schema.yaml"

logger = logging.getLogger(__name__)


def _get_json(exist_ok: bool = False) -> list:
    file_name = "OmniDocBench.json"
    final_path = omni_doc_dir / file_name
    omni_doc_dir.mkdir(parents=True, exist_ok=True)

    if exist_ok and final_path.exists():
        logger.info(
            f"OmniDocBench.json already exists at: {final_path}."
            "Skipping download from Huggingface."
        )
    else:
        hf_hub_download(
            repo_id=omni_doc_repo_id,
            filename=file_name,
            local_dir=omni_doc_dir,
            repo_type="dataset",
        )

    with open(final_path) as f:
        omni_doc = json.load(f)

        if isinstance(omni_doc, list):
            return omni_doc
        else:
            raise ValueError(
                f"Error: Wrong type in OmniDocBench.json. Expected: `list`. Actual: {type(omni_doc)}"
            )


def _filter_images(images: list[str]) -> list[str]:
    filtered = []
    for image in images:
        image_path = image_dir / image
        if not image_path.exists():
            filtered.append(image)

    return filtered


def _get_images(omni_doc, exist_ok: bool = False) -> list[str]:
    images = []

    # research_report could also be interesting, but no multiple choice possible during evaluation
    # also including it does not change the amount of included pages
    wanted_sources = ["academic_literature"]
    wanted_languages = ["english"]

    for instance in omni_doc:
        try:
            info = instance["page_info"]
            attributes = info["page_attribute"]

            lang_ok = attributes["language"] in wanted_languages
            src_ok = attributes["data_source"] in wanted_sources

            if lang_ok and src_ok:
                img = info["image_path"]
                images.append(img)

        except (KeyError, ValueError) as e:
            logger.warning(f"OmniDocItem is malformed: {str(e)}")

    if exist_ok:
        images = _filter_images(images)

    image_dir.mkdir(parents=True, exist_ok=True)
    with open(omni_doc_dir / "included_images.json", "w") as j:
        json.dump(images, j, indent=2)

    snapshot_download(
        repo_id=omni_doc_repo_id,
        local_dir=omni_doc_dir,
        repo_type="dataset",
        allow_patterns=[f"*/{i}" for i in images],
    )

    logger.info(f"Downloaded {len(images)} images from the OmniDocBench benchmark.")
    return images


def _images_to_pdfs(exist_ok: bool = False):
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_dir.glob(pattern="*.jpg"):
        pdf_path = pdf_dir / f"{image_path.stem}.pdf"
        if exist_ok and pdf_path.exists():
            logger.debug(f"Skipping existing PDF for: {image_path.stem}")
        else:
            try:
                pdf = pymupdf.open()
                image = pymupdf.open(filename=image_path)
                rect = image[0].rect
                page = pdf.new_page(width=rect.width, height=rect.height)
                page.insert_image(rect, filename=image_path)

                pdf.save(pdf_path)
                pdf.close()
                logger.debug(f"Successfully created PDF for: {image_path.stem}")
            except Exception as e:
                logger.warning(f"Failed PDF creation for: {image_path.stem}, due to: {str(e)}")


def create_config_files(exist_ok: bool = False):
    if not schema_path.exists():
        logger.error(
            "Missing OmniDocBench config schema "
            f"at: {schema_path}"
        )
        return

    with open(schema_path, "r") as t:
        template = t.read()

    output_dir = CONFIG_DIR / omni_doc_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for parser in Parsers:
        p_name = parser.value
        p_path = output_dir / f"{p_name}.yaml"

        if exist_ok and p_path.exists():
            logger.debug(f"Skipping existing config file for method: {p_name}")
            continue

        dt_path = MD_DIR / p_name / omni_doc_name / "pdfs"

        p_content = template.replace("{{OMNI_DOC_PATH}}", str(gt_path))
        p_content = p_content.replace("{{DT_PATH}}", str(dt_path))

        with open(p_path, "w") as c:
            c.write(p_content)
            logger.debug(f"Created config file for method: {p_name}")


def get_omni_doc_dt_path(parser: Parsers) -> Path:
    return CONFIG_DIR / omni_doc_name / f"{parser.value}.yaml"


def prepare_omni_doc_bench(exist_ok: bool = False):
    omni_doc = _get_json(exist_ok=exist_ok)
    _get_images(omni_doc, exist_ok=exist_ok)
    _images_to_pdfs(exist_ok=exist_ok)
    create_config_files(exist_ok=exist_ok)
    logger.info("Finished setup for OmniDocBench. For usage check the README.md.")


def create_result_table() -> pd.DataFrame:
    """
    Adapted from the OmniDocBench repository.
    Creates a DataFrame containing the benchmark results for every Parser.
    Parsers for which the benchmark has not been run are not included.
    """

    results = []

    matching_algorithm = "quick_match"
    result_dir = OMNI_DOC_PROJECT_PATH / "result"

    metrics = [
        ("text_block", "Edit_dist", r"$\text{Text}^{Edit}\downarrow$"),
        ("table", "TEDS", r"$\text{Table}^{TEDS}\uparrow$"),
        ("table", "TEDS_structure_only", r"$\text{Table}^{S-TEDS^{*}}\uparrow$"),
        ("reading_order", "Edit_dist", r"$\text{Read}^{Edit}\downarrow$")
    ]

    for parser in Parsers:
        res_name = f"{parser.value}_{matching_algorithm}_metric_result.json"
        result_path = result_dir / res_name

        if not result_path.exists():
            logger.warning(
                f"Missing OmniDocBench result for {parser.value} "
                f"at: {result_path}"
            )
            continue

        with open(result_path, 'r') as f:
            result = json.load(f)

        metrics_results = {}

        for category_type, metric, tex_name in metrics:
            if metric == "TEDS" or metric == "TEDS_structure_only":
                if result[category_type]["page"].get(metric):
                    value = result[category_type]["page"][metric]["ALL"] * 100
                else:
                    value = 0
            else:
                value = result[category_type]["all"][metric].get("ALL_page_avg", np.nan)

            metrics_results[tex_name] = value

        parser_name = get_parser_thesis_name(parser)
        result_entry = pd.Series(metrics_results, name=parser_name)
        results.append(result_entry)

    df = pd.DataFrame(results)

    overall_tex = r"$\text{Overall}\uparrow$"

    text_edit_dist = df[metrics[0][2]]
    table_teds = df[metrics[1][2]]
    reading_order_edit_dist = df[metrics[3][2]]

    df[overall_tex] = ((1 - text_edit_dist) * 100 + table_teds + (
        1 - reading_order_edit_dist) * 100) / 3

    return df
