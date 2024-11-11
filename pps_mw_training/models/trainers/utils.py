import gc
import os
import psutil

from keras.backend import clear_session  # type: ignore
from keras.callbacks import Callback  # type: ignore


class MemoryUsageCallback(Callback):
    """Monitor memory usage on epoch begin and end, collect garbage"""

    def memory_usage(self):
        return f"{psutil.Process(os.getpid()).memory_info().rss / 1e6} MB"

    def learning_rate(self):
        return float(self.model.optimizer.learning_rate)

    def info(self):
        return (
            f"memory usage={self.memory_usage()} and "
            f"learning rate={self.learning_rate()}"
        )

    def on_epoch_begin(self, epoch, logs=None):
        print(f"On epoch {epoch + 1} begin: {self.info()}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"On epoch {epoch + 1} end: {self.info()}")
        gc.collect()
        clear_session()
