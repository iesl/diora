import time

from collections import Counter

from diora.logging.accumulator import Accumulator
from diora.logging.configuration import get_logger


class ExperimentLogger(object):
    def __init__(self):
        super(ExperimentLogger, self).__init__()
        self.logger = get_logger()
        self.A = None
        self.c = Counter()

    def str_length_distribution(self):
        result = ''
        keys = sorted(self.c.keys())
        for i, k in enumerate(keys):
            if i > 0:
                result += ' '
            result += '{}:{}'.format(k, self.c[k])
        return result

    def record(self, result):
        if self.A is None:
            self.A = Accumulator()
        A = self.A

        self.c[result['length']] += 1

        for k, v in result.items():
            if 'loss' in k:
                A.record(k, v)
            if 'acc' in k:
                A.record(k, v)

    def log_batch(self, epoch, step, batch_idx, batch_size=1):
        A = self.A
        logger = self.logger

        log_out = 'Epoch/Step/Batch={}/{}/{}'.format(epoch, step, batch_idx)

        for k in A.table.keys():
            if 'loss' in k:
                log_out += ' {}={:.3f}'.format(k, A.get_mean(k))
            if 'acc' in k:
                log_out += ' {}={:.3f}'.format(k, A.get_mean(k))

        logger.info(log_out)

        # Average sentence length from previous batches
        total_length = sum(k * v for k, v in self.c.items())
        total_batches = sum(self.c.values())
        average_length = total_length / total_batches
        logger.info('Average-Length={}'.format(average_length))
        logger.info('Length-Distribution={}'.format(self.str_length_distribution()))

        A.reset()
        self.c.clear()

    def log_epoch(self, epoch, step):
        logger = self.logger
        logger.info('Epoch/Step={}/{} (End-Of-Epoch)'.format(epoch, step))

    def log_eval(self, loss, metric):
        logger = self.logger
        logger.info('Eval Loss={} Metric={}.'.format(loss, metric))
