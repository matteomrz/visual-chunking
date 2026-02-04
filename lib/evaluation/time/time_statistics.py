import json

import pandas as pd

from config import BOUNDING_BOX_DIR
from lib.parsing.model.parsing_result import ParsingMetaData as PmD


def get_time_stats():
    stats = []

    parsing_time_key = PmD.PARSING_TIME.value
    transformation_time_key = PmD.TRANSFORMATION_TIME.value
    parser_key = PmD.PARSER.value
    page_count_key = PmD.PAGE_COUNT.value

    for result_path in BOUNDING_BOX_DIR.rglob("*.json"):
        with open(result_path, "r") as f:
            result = json.load(f)

            if not isinstance(result, dict):
                continue

            metadata = result.get("metadata", {})

            if (
                parser_key in metadata and
                parsing_time_key in metadata and
                transformation_time_key in metadata and
                page_count_key in metadata
            ):
                page_cnt = metadata[page_count_key]
                parsing_pp = metadata[parsing_time_key] / page_cnt
                transformation_pp = metadata[transformation_time_key] / page_cnt

                stats.append({
                    "Method": metadata[parser_key],
                    "Parsing": parsing_pp,
                    "Transformation": transformation_pp
                })

    df = pd.DataFrame(stats)

    comp_table = df.groupby("Method").agg({
        "Parsing": ["mean", "std"],
        "Transformation": ["mean", "std"]
    })

    return comp_table
