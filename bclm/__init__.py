from bclm.pipeline.core import Pipeline
from bclm.models.common.doc import Document
# from bclm.resources.common import download
from bclm._version import __version__, __resources_version__

import logging
logger = logging.getLogger('bclm')

# if the client application hasn't set the log level, we set it
# ourselves to INFO
if logger.level == 0:
    logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)

# also, if the client hasn't added any handlers for this logger
# (or a default handler), we add a handler of our own
#
# client can later do
#   logger.removeHandler(bclm.log_handler)
if not logger.hasHandlers():
    logger.addHandler(log_handler)
