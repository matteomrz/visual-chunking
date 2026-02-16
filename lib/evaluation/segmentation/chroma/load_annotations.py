import json
import logging

import pandas as pd
from pathlib import Path
from typing import TypedDict

from chunking_evaluation.utils import rigorous_document_search
from datasets import tqdm

logger = logging.getLogger(__name__)


def get_offsets_path(corpus_path: Path) -> Path:
    output_dir = corpus_path.parent
    batch_name = corpus_path.stem
    return output_dir / f"{batch_name}_offsets.csv"


class Reference(TypedDict):
    content: str
    start_index: int
    end_index: int


class QAPair(TypedDict):
    question: str
    references: str
    corpus_id: str


def load_annotations(anno_path: Path, corpus_path: Path) -> pd.DataFrame:
    corpus_id = str(corpus_path)
    with open(corpus_path, "r") as f_c:
        corpus = f_c.read()

    offset_path = get_offsets_path(corpus_path)
    offsets = pd.read_csv(offset_path).set_index("pdf_name")

    qa_pairs = []
    error_counter = 0

    with open(anno_path, "r") as f_a:
        for line in tqdm(f_a):
            annotation = json.loads(line)
            question = annotation["question"]
            pdf_name = annotation["pdf_name_in_metadata"]

            doc_offsets = offsets.loc[pdf_name]
            start_idx = doc_offsets["start_index"].item()
            end_idx = doc_offsets["end_index"].item()
            doc_content = corpus[start_idx: end_idx]

            references = []
            for highlight in annotation["highlights"]:
                if highlight["raw_text"] == "":
                    logger.debug("Encountered Highlight with empty text. Skipping...")
                    continue

                result = rigorous_document_search(doc_content, highlight["raw_text"])
                if result is None or not result[0]:
                    error_counter += 1
                    logger.error(
                        "Could not find highlight in document corpus. "
                        f"Searched text: {highlight["raw_text"]}"
                    )
                    continue

                ref = Reference(
                    content=result[0],
                    start_index=start_idx + result[1],
                    end_index=start_idx + result[2]
                )

                references.append(ref)

            if not references:
                logger.error(
                    "Could not find any references in the document corpus. "
                    f"Question: {question}"
                )
                continue

            qa_pair = QAPair(
                question=question,
                references=json.dumps(references),
                corpus_id=corpus_id
            )
            qa_pairs.append(qa_pair)

    logger.info(f"Processed QA pairs with {error_counter} errors.")
    return pd.DataFrame(qa_pairs)
