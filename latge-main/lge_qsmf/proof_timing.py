from contextlib import contextmanager
from contextvars import ContextVar
import time


_active = ContextVar('latge_prover_timer', default=None)


class _ProverTimer:
    def __init__(self):
        self.seconds = self.segment = 0.0
        self.started = None

    def __enter__(self):
        self.token = _active.set(self)
        self.resume()
        return self

    def __exit__(self, *exc):
        self.pause()
        _active.reset(self.token)

    def pause(self):
        if self.started is not None:
            self.segment += time.perf_counter() - self.started
            self.started = None

    def resume(self):
        if self.started is None:
            self.started = time.perf_counter()

    def keep(self):
        running = self.started is not None
        self.pause()
        self.seconds += self.segment
        self.segment = 0.0
        if running:
            self.resume()

    def restart(self):
        self.segment = 0.0
        self.started = time.perf_counter()

    def finish(self):
        self.pause()
        self.keep()
        return self.seconds


def prover_timer():
    return _active.get()


@contextmanager
def excluded_from_prover_time():
    timer = _active.get()
    running = timer is not None and timer.started is not None
    if running:
        timer.pause()
    try:
        yield
    finally:
        if running:
            timer.resume()
