import json

import pandas as pd

from config import PARSING_RESULT_DIR
from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.parsing_result import ParsingMetaData as PmD
from lib.utils.thesis_names import get_parser_thesis_name


def get_time_stats():
    stats = []

    parsing_time_key = PmD.PARSING_TIME.value
    # transformation_time_key = PmD.TRANSFORMATION_TIME.value
    parser_key = PmD.PARSER.value
    page_count_key = PmD.PAGE_COUNT.value

    for result_path in PARSING_RESULT_DIR.rglob("*.json"):
        with open(result_path, "r") as f:
            result = json.load(f)

            if not isinstance(result, dict):
                continue

            metadata = result.get("metadata", {})

            if (
                parser_key in metadata and
                parsing_time_key in metadata and
                # transformation_time_key in metadata and
                page_count_key in metadata
            ):
                page_cnt = metadata[page_count_key]
                parsing_pp = metadata[parsing_time_key] / page_cnt
                # transformation_pp = metadata[transformation_time_key] / page_cnt

                parser = Parsers.get_parser_type(metadata[parser_key])
                parser_name = get_parser_thesis_name(parser)

                stats.append({
                    "Method": parser_name,
                    "seconds per page": parsing_pp,
                    # "Transformation": transformation_pp
                })

    df = pd.DataFrame(stats)

    comp_table = df.groupby("Method", as_index=True).agg({
        "seconds per page": ["mean", "std"],
        # "Transformation": ["mean", "std"]
    })

    comp_table.index.name = None

    return comp_table
