import logging
from pathlib import Path
from typing import Protocol

from lib.parsing.methods.parsers import Parsers
from lib.parsing.model.document_parser import DocumentParser, T
from lib.parsing.model.parsing_result import (
    ParsingBoundingBox, ParsingMetaData as PmD,
    ParsingResult,
    ParsingResultType
)

logger = logging.getLogger(__name__)

dummy = {
    "layout_elements": [
        {
            "category": "page_header",
            "content": "Printed by Anonymous on X. For personal use only. Not approved for distribution. Copyright © 2024 National Comprehensive Cancer Network, Inc., All Rights Reserved.",
            "bbox": {
                "page_number": 1,
                "points": [
                    18,
                    10,
                    680,
                    20
                ]
            }
        },
        {
            "category": "page_header",
            "content": "National\nComprehensive\nCancer\nNetwork®",
            "bbox": {
                "page_number": 1,
                "points": [
                    31,
                    31,
                    222,
                    114
                ]
            }
        },
        {
            "category": "page_header",
            "content": "NCCN Guidelines Version 4.2024\nInvasive Breast Cancer",
            "bbox": {
                "page_number": 1,
                "points": [
                    235,
                    43,
                    584,
                    102
                ]
            }
        },
        {
            "category": "page_header",
            "content": "NCCN Guidelines Index\nTable of Contents\nDiscussion",
            "bbox": {
                "page_number": 1,
                "points": [
                    822,
                    43,
                    967,
                    102
                ]
            }
        },
        {
            "category": "section_header",
            "content": "PRINCIPLES OF RADIATION THERAPY",
            "heading_level": 1,
            "bbox": {
                "page_number": 1,
                "points": [
                    368,
                    146,
                    605,
                    160
                ]
            }
        },
        {
            "category": "section_header",
            "content": "Accelerated Partial Breast Irradiation (APBI)/Partial Breast Irradiation (PBI)",
            "heading_level": 2,
            "bbox": {
                "page_number": 1,
                "points": [
                    31,
                    178,
                    483,
                    192
                ]
            }
        },
        {
            "category": "list_item",
            "content": "- APBI/PBI offers comparable local control to WBRT in selected patients with low-risk early-stage breast cancer. However, the optimal external beam-APBI/PBI technique/fractionation for minimizing long-term cosmesis effects has not been determined.",
            "bbox": {
                "page_number": 1,
                "points": [
                    31,
                    197,
                    915,
                    237
                ]
            }
        },
        {
            "category": "text",
            "content": "Patients are encouraged to participate in clinical trials.",
            "bbox": {
                "page_number": 1,
                "points": [
                    48,
                    237,
                    360,
                    249
                ]
            }
        },
        {
            "category": "text",
            "content": "The NCCN Panel recommends APBI/PBI for any patient with no BRCA 1/2 mutations meeting the criteria outlined in the 2016 ASTRO consensus statement for guidance on APBI/PBI use.",
            "bbox": {
                "page_number": 1,
                "points": [
                    31,
                    256,
                    829,
                    281
                ]
            }
        },
        {
            "category": "text",
            "content": "According to the 2016 ASTRO criteria, patients aged ≥50 years are \"suitable\" for APBI/PBI if they have:",
            "bbox": {
                "page_number": 1,
                "points": [
                    48,
                    281,
                    747,
                    296
                ]
            }
        },
        {
            "category": "list_item",
            "content": "- Invasive ductal carcinoma measuring ≤2 cm (pT1 disease) with negative margin widths of ≥2 mm, no LVI, and ER-positive tumors",
            "bbox": {
                "page_number": 1,
                "points": [
                    53,
                    296,
                    917,
                    335
                ]
            }
        },
        {
            "category": "text",
            "content": "or",
            "bbox": {
                "page_number": 1,
                "points": [
                    53,
                    335,
                    65,
                    347
                ]
            }
        },
        {
            "category": "list_item",
            "content": "- Low/intermediate nuclear grade, screening-detected DCIS measuring size ≤2.5 cm with negative margin widths of ≥3 mm.",
            "bbox": {
                "page_number": 1,
                "points": [
                    53,
                    347,
                    856,
                    367
                ]
            }
        },
        {
            "category": "list_item",
            "content": "- RT dosing:",
            "bbox": {
                "page_number": 1,
                "points": [
                    31,
                    381,
                    98,
                    396
                ]
            }
        },
        {
            "category": "table",
            "content": "| Regimen | Method | Reference |\n|---|---|---|\n| 30 Gy/5 fractions QOD (preferred) | External beam RT (EBRT) ᵉ | Livi L, Meattini I, Marrazzo L, et al. Accelerated partial breast irradiation using intensity-modulated radiotherapy versus whole breast irradiation: 5-year survival analysis of a phase 3 randomised controlled trial. Eur J Cancer 2015;51:451-463.<br>Meattini I, Marrazzo L, Saieva C, et al. Accelerated partial-breast irradiation compared with whole-breast irradiation for early breast cancer: Long-term results of the randomized phase III APBI-IMRT-Florence Trial. J Clin Oncol 2020;38:4175-4183. |\n| 40 Gy/15 fractions | EBRT | Coles CE, Griffin CL, Kirby AM, et al. Partial-breast radiotherapy after breast conservation surgery for patients with early breast cancer (UK IMPORT LOW trial): 5-year results from a multicentre, randomised, controlled, phase 3, non-inferiority trial. Lancet 2017;390:1048-1060. |\n| 34 Gy/10 fractions BID | Balloon/ Interstitial | Vicini FA, Cecchini RS, White JR, et al. Long-term primary results of accelerated partial breast irradiation after breast-conserving surgery for early-stage breast cancer: a randomised, phase 3, equivalence trial. Lancet 2019;394:2155-2164. |\n| 38.5 Gy/10 fractions BID | EBRT | Whelan TJ, Julian JA, Berrang TS, et al. External beam accelerated partial breast irradiation versus whole breast irradiation after breast conserving surgery in women with ductal carcinoma in situ and node-negative breast cancer (RAPID): a randomised controlled trial. Lancet 2019;394:2165-2172. |",
            "bbox": {
                "page_number": 1,
                "points": [
                    44,
                    421,
                    954,
                    743
                ]
            }
        },
        {
            "category": "text",
            "content": "ᵉ The protocol mandated IMRT.",
            "bbox": {
                "page_number": 1,
                "points": [
                    28,
                    875,
                    203,
                    892
                ]
            }
        },
        {
            "category": "text",
            "content": "Note: All recommendations are category 2A unless otherwise indicated.",
            "bbox": {
                "page_number": 1,
                "points": [
                    32,
                    914,
                    388,
                    938
                ]
            }
        },
        {
            "category": "page_footer",
            "content": "Version 4.2024, 07/03/24 © 2024 National Comprehensive Cancer Network® (NCCN®), All rights reserved. NCCN Guidelines® and this illustration may not be reproduced in any form without the express written permission of NCCN.",
            "bbox": {
                "page_number": 1,
                "points": [
                    28,
                    960,
                    794,
                    971
                ]
            }
        },
        {
            "category": "page_footer",
            "content": "BINV-I\n3 OF 3",
            "bbox": {
                "page_number": 1,
                "points": [
                    926,
                    929,
                    969,
                    959
                ]
            }
        }
    ]
}


class VLMParser(DocumentParser):
    """Base class that uses a single-stage VLM for document parsing."""

    module = Parsers.VLM

    label_mapping = {
        t.value: t
        for t in ParsingResultType
    }

    def _parse(self, file_path: Path, options: dict = None) -> dict:
        """Implemented by specific VLMParser."""
        return dummy

    def _get_md(self, raw_result: dict, file_path: Path) -> str:
        return "# TEST"

    def _transform(self, raw_result: dict) -> ParsingResult:
        root = ParsingResult.root()

        seen_elements: dict[ParsingResultType, int] = {}
        for elem in raw_result.get("layout_elements", []):
            try:
                content = elem["content"]
                raw_type = elem["category"]

            except KeyError:
                logger.warning(f"Skipping malformed element: {elem}")
                continue

            elem_type = self._get_element_type(raw_type)

            type_cnt = seen_elements.get(elem_type, 0)
            elem_id = f"{elem_type.value}_{type_cnt}"
            seen_elements[elem_type] = type_cnt + 1

            raw_box = elem.get("bbox", {})
            try:
                page_num = raw_box["page_number"]
                points = raw_box["points"]

                if len(points) != 4:
                    raise ValueError()

                # Coordinates from gemini are always 0-1000
                points = [p / 1000 for p in points]

                if any(p > 1.0 or p < 0.0 for p in points):
                    raise ValueError()

            except (KeyError, ValueError):
                logger.warning(f"Malformed bounding box: {raw_box}")
                points = [0.0, 0.0, 0.0, 0.0]
                page_num = 1

            box = ParsingBoundingBox(
                page=page_num,
                left=points[0],
                top=points[1],
                right=points[2],
                bottom=points[3]
            )

            res = ParsingResult(
                id=elem_id,
                type=elem_type,
                content=content,
                parent=root,
                geom=[box]
            )

            if elem_type == ParsingResultType.SECTION_HEADER:
                heading_level = elem.get("heading_level", 1)
                res.metadata[PmD.HEADER_LEVEL.value] = heading_level

            root.children.append(res)

        return root
