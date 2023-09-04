import logging
import os
import sys
from logging import Logger
from time import perf_counter


def get_logger(logdir, console: bool, file: bool) -> Logger:
    logger = logging.getLogger(logdir)

    formatter = logging.Formatter("%(message)s")
    logger.setLevel(logging.INFO)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if file:
        file_handler = logging.FileHandler(
            os.path.join(logdir, "model.log"), encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.terminator = ""
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.terminator = ""
        logger.addHandler(console_handler)

    return logger


class Catchtime:
    def __init__(self):
        self.time = float("inf")

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
