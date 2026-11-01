import asyncio

import pytest

from cholescope.control_stream import ActionStream


class Clock:
    def __init__(self):
        self.now = 0.
        self.waiters = []

    async def sleep(self, delay):
        if delay > 0:
            future = asyncio.get_running_loop().create_future()
            self.waiters.append((self.now + delay, future))
            await future
        else:
            await asyncio.sleep(0)

    async def tick(self, now):
        self.now = now
        for deadline, future in self.waiters:
            if deadline <= now + 1e-9 and not future.done():
                future.set_result(None)
        self.waiters = [(at, future) for at, future in self.waiters if not future.done()]
        for _ in range(3):
            await asyncio.sleep(0)


async def test_sampler_keeps_latest_chunk_and_reports_overload_without_backdating():
    clock = Clock()
    sampled = []

    def sample(now):
        sampled.append(now)
        return [now]

    stream = ActionStream(sample, lambda: True, 4, clock=lambda: clock.now, sleep=clock.sleep)
    stream.start()
    try:
        for index in range(10):
            await clock.tick(index / 10)
        assert stream.queue.qsize() == 4
        assert stream.skipped == 6
        rows = await stream.chunk(4)
        assert [row.at for row in rows] == [.6, .7, .8, .9]
        assert [row.action for row in rows] == [[.6], [.7], [.8], [.9]]
        # A stalled event loop exports one current pose, not 90 stale ticks.
        await clock.tick(10.)
        assert sampled[-1] == 10.
        assert len(sampled) == 11
        assert stream.queue.qsize() == 1
        await clock.tick(10.1)
        assert sampled[-1] == 10.1
    finally:
        await stream.close()
    assert stream.task.done()


async def test_stop_wakes_partial_chunk_consumer_and_discards_pending_rows():
    clock = Clock()
    stream = ActionStream(lambda now: [now], lambda: True, 4,
                          clock=lambda: clock.now, sleep=clock.sleep)
    stream.start()
    consumer = asyncio.create_task(stream.chunk(4))
    await clock.tick(0.)
    assert not consumer.done()
    stream.stop()
    assert await asyncio.wait_for(consumer, .5) is None
    await stream.close()
    assert stream.task.done()


async def test_sampling_failure_reaches_consumer_instead_of_hanging():
    def broken(now):
        raise ValueError("sample failure")

    stream = ActionStream(broken, lambda: True, 4)
    stream.start()
    try:
        with pytest.raises(ValueError, match="sample failure"):
            await asyncio.wait_for(stream.chunk(4), .5)
    finally:
        await stream.close()
