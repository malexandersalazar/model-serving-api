import os
import logging
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler

class LoggerService():
    def __init__(self, log_level: str, log_format: str, log_dir: str):
        self.logger = logging.getLogger("app")
        self.log_level = getattr(logging, log_level.upper())
        self.log_format = log_format
        self.log_dir = log_dir
        self.__configure_logger()

    def __configure_logger(self):
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Configure file handler
        log_file = os.path.join(
            self.log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1024 * 1024, backupCount=10  # 1MB
        )
        file_handler.setLevel(self.log_level)

        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)

        # Create formatter
        formatter = logging.Formatter(self.log_format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(self.log_level)

    def info(self, message: str, extra: Optional[dict] = None):
        self.logger.info(message, extra=extra)

    def error(self, message: str, extra: Optional[dict] = None):
        self.logger.error(message, extra=extra)

    def warning(self, message: str, extra: Optional[dict] = None):
        self.logger.warning(message, extra=extra)

    def debug(self, message: str, extra: Optional[dict] = None):
        self.logger.debug(message, extra=extra)

    def critical(self, message: str, extra: Optional[dict] = None):
        self.logger.critical(message, extra=extra)
