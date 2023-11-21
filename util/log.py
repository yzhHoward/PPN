import os
import sys
import time
import logging


def init_logging(log_root, models_root=None):
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    if models_root is not None:
        handler_file = logging.FileHandler(
            os.path.join(models_root, "training.log"))
        handler_file.setFormatter(formatter)
        log_root.addHandler(handler_file)
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_stream)


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logging(object):
    def __init__(self, total_epoch):
        self.time_start = time.time()
        self.total_epoch: int = total_epoch

        self.init = False
        self.tic = 0
        self.cnt = 0

    def log_batch(self, epoch, batch, loss):
        logging.info('Epoch %d Batch %d: Train Loss = %.4f' %
                     (epoch, batch, loss))

    def log_epoch(self, epoch, loss, valid_loss, roc, fold=None):
        if self.init:
            time_now = (time.time() - self.time_start) / 60
            self.cnt += 1
            time_total = time_now / (self.cnt / self.total_epoch)
            time_for_end = time_total - time_now
            if fold is not None:
                msg = "Fold %d, Epoch %d: Loss = %.4f Valid Loss = %.4f roc = %.4f %1.f hours %2.f mins" % (fold,
                                                                                                            epoch, loss, valid_loss, roc, time_for_end // 60, time_for_end % 60
                                                                                                            )
            else:
                msg = "Epoch %d: Loss = %.4f Valid Loss = %.4f roc = %.4f Required: %1.f hours %2.f mins" % (
                    epoch, loss, valid_loss, roc, time_for_end // 60, time_for_end % 60
                )
            logging.info(msg)
            self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
