import logging.config
import yaml

from config import LIB_DIR

# Set config for logging for all modules in lib
with open(LIB_DIR / "logging.yaml", "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    logger.debug("Successfully initialized Logging.")
