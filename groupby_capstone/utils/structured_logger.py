"""Structured Logger

 This class provides helper functions to log messages in a helpful form.

 NOTE: this is copied from the etl-vistara project

 Proper usage:
 #at top of project file:
 from .utils.structured_logger import StructuredLogger
 __all__ = ['CostTrendsReportGenerator']
structured_logger = StructuredLogger()

#within project file at location of logging:
structured_logger.logInfo("Starting Report Generation", __name__)
#or for error handling
structured_logger.logFatal(error, __name__)
"""
import json
import datetime
import logging
from functools import singledispatch
from structlog import get_logger


class StructuredLogger():

    @singledispatch
    def to_serializable(self, val):
        """Used by default."""
        return str(val)

    @to_serializable.register(datetime.datetime)
    def ts_datetime(self, val):
        """Used if *val* is an instance of datetime."""
        return val.isoformat() + "Z"

    def info(self, info_message, name, properties=None, messageTemplate=None):
        Logger = get_logger(name)
        if messageTemplate is None:
            messageTemplate = "{info_message}"

        if properties is None:
            Logger.info(
                MessageTemplate=messageTemplate,
                RenderedMessage=f'{info_message}',
                Level="Information",
                SourceContext=name,
                Timestamp=f'{self.ts_datetime(datetime.datetime.utcnow())}'
            )
        else:
            Logger.info(
                MessageTemplate=messageTemplate,
                RenderedMessage=f'{info_message}',
                Properties=properties,
                Level="Information",
                SourceContext=name,
                Timestamp=f'{self.ts_datetime(datetime.datetime.utcnow())}'
            )

    def error(self, error_message, name):
        logger = get_logger(name)
        logger.error(
            MessageTemplate='Encountered an error: {error_message}',
            RenderedMessage=f'Encountered an error: {error_message}',
            Properties={"error": {error_message}},
            Level="Error",
            SourceContext=name,
            Timestamp=f'{self.ts_datetime(datetime.datetime.utcnow())}'
        )

    def debug(self, debug_message, name, properties=None, messageTemplate=None):
        Logger = get_logger(name)
        if messageTemplate is None:
            messageTemplate = "{debug_message}"

        if properties is None:
            Logger.debug(
                MessageTemplate=messageTemplate,
                RenderedMessage=f'{debug_message}',
                Level="Debug",
                SourceContext=name,
                Timestamp=f'{self.ts_datetime(datetime.datetime.utcnow())}'
            )
        else:
            Logger.debug(
                MessageTemplate=messageTemplate,
                RenderedMessage=f'{debug_message}',
                Properties=properties,
                Level="Debug",
                SourceContext=name,
                Timestamp=f'{self.ts_datetime(datetime.datetime.utcnow())}'
            )

    def warn(self, warning_message, name, properties=None, messageTemplate=None):
        logger = get_logger(name)
        if messageTemplate is None:
            messageTemplate = 'Warning: {warning_message}'

        if properties is None:
            logger.warning(
                MessageTemplate=messageTemplate,
                RenderedMessage=f'Warning: {warning_message}',
                Level="Warning",
                SourceContext=name,
                Timestamp=f'{self.ts_datetime(datetime.datetime.utcnow())}'
            )
        else:
            logger.warning(
                MessageTemplate='Warning: {warning_message}',
                RenderedMessage=f'Warning: {warning_message}',
                Properties=properties,
                Level="Warning",
                SourceContext=name,
                Timestamp=f'{self.ts_datetime(datetime.datetime.utcnow())}'
            )

    def fatal(self, fatal_message, name):
        properties = {"fatal_error": {fatal_message}}
        fatal_log = {
            "MessageTemplate": "Encountered a fatal error: {fatal_message}",
            "RenderedMessage": f'Encountered a fatal error: {fatal_message}',
            "Properties": properties,
            "Level": "fatal",
            "SourceContext": f'{name}',
            "Timestamp": self.ts_datetime(datetime.datetime.utcnow())
        }
        jsonDump = json.dumps(fatal_log, default=self.to_serializable)
        logging.fatal(jsonDump)
