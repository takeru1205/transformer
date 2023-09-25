from functools import wraps
from logging import getLogger

logger = getLogger("Log").getChild(__name__)


def log_inout(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logger.debug(f"enter: {f.__name__}")
        res = f(*args, **kwargs)
        logger.debug(f"exit: {f.__name__}")
        return res

    return wrapper


if __name__ == "__main__":

    @log_inout
    def sample_func(text):
        print(text)
        text += "logging"
        print(text)
        return text

    import logging

    logging.basicConfig(level=logging.DEBUG)
    text_arg = "sample"
    assert sample_func(text_arg) == text_arg + "logging"
    print(sample_func.__name__)
