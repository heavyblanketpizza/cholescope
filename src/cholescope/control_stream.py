"""Bounded live action sampling independent of model inference and playback."""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSample:
    at: float
    action: list[float]


class ActionStream:
    """Sample at 10 Hz, retaining at most one pending chunk of recent poses.

    Slow inference drops the oldest pending snapshots, never key events or the
    controller's integrated pose. A stalled event loop skips missed deadlines
    instead of creating a burst of backdated actions. The consumer still sends
    exactly one ordered, fixed-size chunk per model call.
    """

    def __init__(self, sample, active, capacity, *, clock=time.monotonic, sleep=asyncio.sleep):
        self.sample, self.active = sample, active
        self.clock, self.sleep = clock, sleep
        self.queue = asyncio.Queue(maxsize=capacity)
        self.skipped = 0
        self.error = None
        self.task = None
        self.stopped = False

    def start(self):
        self.task = asyncio.create_task(self._run())

    def stop(self):
        self.stopped = True
        if self.task and not self.task.done():
            self.task.cancel()
        self._end()

    def _end(self):
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(None)

    async def close(self):
        self.stop()
        if self.task:
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def _run(self):
        deadline = self.clock()
        try:
            while not self.stopped and self.active():
                await self.sleep(max(0., deadline - self.clock()))
                if self.stopped or not self.active():
                    break
                now = self.clock()
                row = ActionSample(now, self.sample(now))
                if self.queue.full():
                    self.queue.get_nowait()
                    self.skipped += 1
                self.queue.put_nowait(row)
                deadline += .1
                after_sample = self.clock()
                if deadline <= after_sample:
                    deadline += (math.floor((after_sample - deadline) / .1) + 1) * .1
        except Exception as error:
            self.error = error
        finally:
            self._end()

    async def chunk(self, count):
        rows = []
        for _ in range(count):
            row = await self.queue.get()
            if row is None:
                if self.error:
                    raise self.error
                return None
            rows.append(row)
        return rows if self.active() and not self.stopped else None
