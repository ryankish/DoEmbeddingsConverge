import logging

def green(*args, log_level=logging.INFO):
    message = ' '.join(map(str, args))
    print(f"\033[92m{message}\033[0m")
    if log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)

def cyan(*args, log_level=logging.INFO):
    message = ' '.join(map(str, args))
    print(f"\033[96m{message}\033[0m")
    if log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)


def orange(*args):
    print(f"\033[93m{' '.join(map(str, args))}\033[0m")

def green(*args):
    print(f"\033[92m{' '.join(map(str, args))}\033[0m")

def red(*args):
    print(f"\033[91m{' '.join(map(str, args))}\033[0m")
