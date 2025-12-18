import logging.config
import yaml
import config

# Set config for logging for all modules in lib
with open(config.LIB_DIR / "logging.yaml", "r") as f:
    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)

    logger = logging.getLogger(__name__)
    logger.debug("Successfully initialized Logging.")
