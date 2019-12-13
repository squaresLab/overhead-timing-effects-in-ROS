# -*- coding: utf-8 -*-
__all__ = ('CircleIntBuffer', 'wait_till_open')

from contextlib import closing
from typing import Tuple
from threading import Lock
from timeit import default_timer as timer
import socket
import time

import attr


def wait_till_open(port: int,
                   timeout_seconds: int,
                   *,
                   interval: float = 0.5
                   ) -> None:
    """Blocks until either a given local port is open or a timeout expires."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        time_start = timer()
        time_stop = time_start + timeout_seconds
        while timer() < time_stop:
            if s.connect_ex(('localhost', port)) == 0:
                return
            time.sleep(interval)
    m = f"unable to reach port [{port}] after {timeout_seconds} seconds"
    raise TimeoutError(m)


@attr.s(auto_attribs=True)
class CircleIntBuffer:
    """An end-inclusive ring buffer."""
    start: int
    stop: int
    _next: int = attr.ib(init=False)
    _lock: Lock = attr.ib(init=False, factory=Lock)

    def __attrs_post_init__(self) -> None:
        self._size = self.stop - self.start + 1
        self._next = self.start

    def __next__(self) -> int:
        with self._lock:
            n = self._next
            self._next += 1
            if self._next == self.stop + 1:
                self._next = self.start
            return n

    def take(self, n: int) -> Tuple[int, ...]:
        assert n <= self._size
        return tuple(next(self) for i in range(n))
