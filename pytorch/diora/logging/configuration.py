import os
import logging

from diora.utils.fs import mkdir_p


LOGGING_NAMESPACE = 'diora'


def configure_experiment(experiment_path, rank=None):
    mkdir_p(experiment_path)
    if rank is None:
        log_file = os.path.join(experiment_path, 'experiment.log')
    else:
        log_file = os.path.join(experiment_path, 'experiment.log.{}'.format(rank))
    configure_logger(log_file)


def configure_logger(log_file):
    """
    Simple logging configuration.
    """

    # Create logger.
    logger = logging.getLogger(LOGGING_NAMESPACE)
    logger.setLevel(logging.INFO)

    # Create file handler.
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Also log to console.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # HACK: Weird fix that counteracts other libraries (i.e. allennlp) modifying
    # the global logger.
    if len(logger.parent.handlers) > 0:
        logger.parent.handlers.pop()

    return logger


def get_logger():
    return logging.getLogger(LOGGING_NAMESPACE)
