"""Continuous acquisition; slow consumers only ever receive the newest window."""

from collections import deque
from threading import Condition, Event, Thread
import time

import numpy as np


class LatestSamples:
    def __init__(self, source, capacity=600):
        self.source = source
        self.rows = deque(maxlen=capacity)
        self.condition = Condition()
        self.stop = Event()
        self.sequence = 0
        self.received_at = 0.0
        self.started_at = time.monotonic()
        self.error = None
        self.thread = Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read(self):
        try:
            while not self.stop.is_set():
                row = self.source.read_sample(max_wait_seconds=2.0)
                with self.condition:
                    self.rows.append(row)
                    self.sequence += 1
                    self.received_at = time.monotonic()
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def snapshot(self, after=0, timeout=0.05):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > after or self.error, timeout)
            if self.error:
                raise self.error
            if self.sequence <= after:
                return None
            return np.asarray(self.rows, dtype=np.float32), self.sequence, self.received_at

    def close(self):
        self.stop.set()
        self.source.close()
        self.thread.join(timeout=1.2)
