import settings

import logging
import inspect
from pythonjsonlogger import jsonlogger


class Logger:
    def __init__(self, log_file):
        self.logger = logging.getLogger(settings.APP_NAME)
        self.logger.setLevel(logging.INFO)

        self.dbLogger = logging.getLogger("sqlalchemy.engine")
        self.dbLogger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(f"{settings.LOG_DIR}/{log_file}")
        file_handler.setLevel(logging.INFO)

        dbFileHandler = logging.FileHandler(settings.DATABASE_LOG)
        dbFileHandler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = jsonlogger.JsonFormatter(
            fmt="%(levelname)s %(asctime)s %(name) %(message)",
            datefmt="%Y-%m-%d %H:%M.%S",
            json_ensure_ascii=True,
        )

        file_handler.setFormatter(formatter)
        dbFileHandler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.dbLogger.addHandler(dbFileHandler)

    def log(self, message, severity=""):
        frame = inspect.currentframe().f_back
        function_name = frame.f_code.co_name
        file_name = frame.f_code.co_filename
        line_number = frame.f_lineno

        extra_info = {
            'function': f'{function_name}()',
            'file': file_name.split('/')[-1],
            'line': line_number
        }

        if not severity or severity == "info":
            self.logger.info(message, extra=extra_info)
        elif severity == "debug":
            self.logger.debug(message, extra=extra_info)
        elif severity == "warning":
            self.logger.warning(message, extra=extra_info)
        elif severity == "error":
            self.logger.error(message, extra=extra_info)
        elif severity == "critical":
            self.logger.critical(message, extra=extra_info)
