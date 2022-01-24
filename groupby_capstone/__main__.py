import logging
import os
import sys

import structlog
import traceback

from groupby_capstone.utils.structured_logger import StructuredLogger
from .app import run

logger = StructuredLogger()


def setup_logger(log_level='info'):
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    root_log_level = log_levels.get(log_level.lower(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(root_log_level)

    logging.basicConfig(
        level=root_log_level,
        format="%(message)s",
        handlers=[handler],
    )

    structlog.configure(
        processors=[structlog.processors.JSONRenderer()],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def main(cmd_args, *args, **kwargs):
    if os.environ.get('log_level'):
        setup_logger(os.environ['log_level'])
    else:
        setup_logger()

    command = "webserver"
    if cmd_args:
        command = cmd_args[0]

    run(command)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as error:
        logger.fatal(error, __name__)
        logger.fatal(traceback.format_exc(), __name__)
        sys.exit(1)
