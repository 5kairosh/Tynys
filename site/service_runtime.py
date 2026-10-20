"""Bounded, process-local admission and a single inference worker.

HTTP timeouts cannot interrupt native TensorFlow work. A detached worker keeps
both permits until it really finishes; timed-out clients cannot grow a hidden
executor queue. No filenames, images, scores or user identifiers are recorded.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

from fastapi import HTTPException
from starlette.responses import JSONResponse


@dataclass(frozen=True)
class Limits:
    capacity: int = 3  # Includes uploading, waiting and detached native work.
    queue_seconds: float = 20
    predict_seconds: float = 90
    report_seconds: float = 180
    retry_seconds: int = 5


class Lease:
    def __init__(self, runtime):
        self.runtime = runtime
        self.future = None
        self.closed = False

    def release(self):
        if self.closed:
            return
        if self.future is not None and not self.future.done():
            return  # Completion callback owns cleanup after timeout/disconnect.
        self.closed = True
        self.runtime.admitted -= 1


class InferenceRuntime:
    def __init__(self, limits=None):
        self.limits = limits or Limits()
        self.admitted = self.active = self.waiting = 0
        self.counters = {key: 0 for key in ('accepted', 'busy_rejections', 'queue_timeouts',
                                           'processing_timeouts', 'completed', 'failed')}
        self.started = time.monotonic()
        self.worker = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tynys-inference')
        self.futures = set()
        self.stopping = False

    def reserve(self):
        # Called only on the ASGI event loop, with no await between check/update.
        if self.stopping or self.admitted >= self.limits.capacity:
            self.counters['busy_rejections'] += 1
            return None
        self.admitted += 1
        self.counters['accepted'] += 1
        return Lease(self)

    def snapshot(self):
        return {'policy': 'bounded_inference_v1', 'capacity': self.limits.capacity,
                'active': self.active, 'waiting': self.waiting, 'admitted': self.admitted,
                'accepting': not self.stopping and self.admitted < self.limits.capacity,
                'queue_timeout_seconds': self.limits.queue_seconds,
                'predict_timeout_seconds': self.limits.predict_seconds,
                'report_timeout_seconds': self.limits.report_seconds,
                'retry_after_seconds': self.limits.retry_seconds,
                'uptime_seconds': round(time.monotonic() - self.started),
                'counters': dict(self.counters)}

    async def run(self, lease, function, *args, report=False):
        self.waiting += 1
        try:
            try:
                await asyncio.wait_for(self.worker.acquire(), self.limits.queue_seconds)
            except TimeoutError:
                self.counters['queue_timeouts'] += 1
                raise HTTPException(503, 'Талдау кезегін күту уақыты аяқталды. Біраздан соң қайта басыңыз.',
                                    headers={'Retry-After': str(self.limits.retry_seconds)}) from None
        finally:
            self.waiting -= 1
        self.active += 1
        try:
            future = asyncio.get_running_loop().run_in_executor(self.executor, function, *args)
        except BaseException:
            self.active -= 1
            self.worker.release()
            raise
        lease.future = future
        self.futures.add(future)

        def finished(done):
            # Runs on the event loop, even when the HTTP caller has gone away.
            exception = None if done.cancelled() else done.exception()
            self.counters['failed' if done.cancelled() or exception else 'completed'] += 1
            self.active -= 1
            self.worker.release()
            self.futures.discard(done)
            lease.release()

        future.add_done_callback(finished)
        seconds = self.limits.report_seconds if report else self.limits.predict_seconds
        try:
            return await asyncio.wait_for(asyncio.shield(future), seconds)
        except TimeoutError:
            if future.done():
                # A worker's own TimeoutError is not our response deadline.
                return future.result()
            self.counters['processing_timeouts'] += 1
            raise HTTPException(504, 'Талдау уақыты аяқталды. Нәтиже алынбады. Біраздан соң қайта басыңыз.',
                                headers={'Retry-After': str(self.limits.retry_seconds)}) from None

    async def close(self):
        self.stopping = True
        if self.futures:
            await asyncio.gather(*(asyncio.shield(f) for f in tuple(self.futures)), return_exceptions=True)
        self.executor.shutdown(wait=False)


class InferenceAdmission:
    """Reject excess jobs before buffering or parsing their multipart bodies."""
    def __init__(self, app, runtime):
        self.app, self.runtime = app, runtime

    async def __call__(self, scope, receive, send):
        if (scope['type'] != 'http' or scope.get('method') != 'POST'
                or scope.get('path', '').rstrip('/') not in ('/api/predict', '/api/report')):
            return await self.app(scope, receive, send)
        lease = self.runtime.reserve()
        if lease is None:
            response = JSONResponse({'detail': 'Сервер бос емес. 5 секундтан кейін қайта басыңыз.',
                                     'code': 'server_busy'}, status_code=429,
                                    headers={'Retry-After': str(self.runtime.limits.retry_seconds)})
            return await response(scope, receive, send)
        scope.setdefault('state', {})['inference_lease'] = lease
        try:
            await self.app(scope, receive, send)
        finally:
            lease.release()
