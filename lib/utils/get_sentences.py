import logging

import nltk

logger = logging.getLogger(__name__)


def get_sentences(text: str):
    return nltk.sent_tokenize(text)


PUNKT_TAB_NAME = "punkt_tab"


def setup_nltk():
    try:
        nltk.data.find(f"tokenizers/{PUNKT_TAB_NAME}.zip")
    except LookupError:
        logger.info(f"Downloading '{PUNKT_TAB_NAME}' for NLTK...")
        nltk.download(PUNKT_TAB_NAME)
