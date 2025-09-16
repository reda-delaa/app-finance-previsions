import warnings, logging

_emitted_once = set()

def warn_once(logger: logging.Logger, key: str, message: str, wcat=UserWarning):
    if key in _emitted_once:
        return
    _emitted_once.add(key)
    warnings.warn(message, wcat)
    logger.warning(message)
