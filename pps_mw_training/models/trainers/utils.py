import gc
import os
import psutil

from tensorflow.keras import backend as k  # type: ignore
from tensorflow.keras.callbacks import Callback  # type: ignore


class MemoryUsageCallback(Callback):
    """Monitor memory usage on epoch begin and end, collect garbage"""

    def pss(self):
        return f"{psutil.Process(os.getpid()).memory_info().rss / 1e6} MB"

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}, memory usage on epoch begin: {self.pss()}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}, memory usage on epoch end:  {self.pss()}.")
        gc.collect()
        k.clear_session()
