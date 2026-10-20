"""Deterministic overload/timeout tests; synchronization, not medical accuracy."""
import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI, Request

from service_runtime import InferenceAdmission, InferenceRuntime, Limits
from upload_memory import UploadMemoryLimit, PREDICT_BODY


def harness(function, limits=None):
    runtime = InferenceRuntime(limits)
    app = FastAPI()
    app.add_middleware(UploadMemoryLimit)
    app.add_middleware(InferenceAdmission, runtime=runtime)

    @app.post('/api/predict')
    @app.post('/api/report')
    async def predict(request: Request):
        return await runtime.run(request.state.inference_lease, function,
                                 report=request.url.path.endswith('report'))

    @app.get('/api/health')
    async def health():
        return runtime.snapshot()

    return app, runtime


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(.001)


def test_shared_worker_bounds_predict_and_report_without_blocking_health():
    async def scenario():
        started, release = threading.Event(), threading.Event()
        calls = []

        def work():
            calls.append(len(calls))
            started.set()
            assert release.wait(3)
            return {'ok': True}

        app, runtime = harness(work)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                first = asyncio.create_task(client.post('/api/predict'))
                await until(started.is_set)
                second = asyncio.create_task(client.post('/api/report'))
                third = asyncio.create_task(client.post('/api/predict'))
                await until(lambda: runtime.waiting == 2)
                health = await asyncio.wait_for(client.get('/api/health'), .5)
                assert health.json()['active'] == 1 and health.json()['admitted'] == 3
                overflow = await client.post('/api/predict', content=b'not read')
                assert overflow.status_code == 429 and overflow.headers['retry-after'] == '5'
                assert calls == [0]
                release.set()
                assert [r.status_code for r in await asyncio.gather(first, second, third)] == [200] * 3
                assert len(calls) == 3
                assert runtime.admitted == runtime.active == runtime.waiting == 0
        finally:
            release.set()
            await runtime.close()

    asyncio.run(scenario())


def test_capacity_rejection_does_not_read_body():
    async def scenario():
        runtime = InferenceRuntime(Limits(capacity=1))
        lease = runtime.reserve()
        messages = []

        async def forbidden(*args):
            pytest.fail('Rejected request must not reach receive or application.')

        async def send(message):
            messages.append(message)

        try:
            await InferenceAdmission(forbidden, runtime)(
                {'type': 'http', 'method': 'POST', 'path': '/api/predict'}, forbidden, send)
            assert messages[0]['status'] == 429
        finally:
            lease.release()
            await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('late_failure', [False, True])
def test_processing_timeout_keeps_slot_until_native_work_finishes(late_failure):
    async def scenario():
        release = threading.Event()

        def work():
            assert release.wait(3)
            if late_failure:
                raise ValueError('Late native failure')
            return {'ok': True}

        app, runtime = harness(work, Limits(capacity=1, predict_seconds=.02))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await client.post('/api/predict')
                assert response.status_code == 504
                assert runtime.active == runtime.admitted == 1
                assert (await client.post('/api/predict')).status_code == 429
                assert (await client.get('/api/health')).status_code == 200
                release.set()
                await until(lambda: runtime.admitted == 0)
                assert runtime.counters['processing_timeouts'] == 1
                assert runtime.counters['failed' if late_failure else 'completed'] == 1
        finally:
            release.set()
            await runtime.close()

    asyncio.run(scenario())


def test_queue_timeout_does_not_submit_an_extra_job():
    async def scenario():
        started, release = threading.Event(), threading.Event()
        calls = []

        def work():
            calls.append(1)
            started.set()
            assert release.wait(3)
            return {'ok': True}

        app, runtime = harness(work, Limits(queue_seconds=.02))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                first = asyncio.create_task(client.post('/api/predict'))
                await until(started.is_set)
                response = await client.post('/api/report')
                assert response.status_code == 503 and response.headers['retry-after'] == '5'
                assert runtime.admitted == 1 and runtime.waiting == 0 and len(calls) == 1
                release.set()
                assert (await first).status_code == 200
        finally:
            release.set()
            await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_active', [True, False])
def test_cancellation_cannot_release_running_worker_early(cancel_active):
    async def scenario():
        started, release = threading.Event(), threading.Event()

        def work():
            started.set()
            assert release.wait(3)
            return True

        runtime = InferenceRuntime()
        first_lease = runtime.reserve()
        first = asyncio.create_task(runtime.run(first_lease, work))
        await until(started.is_set)
        second_lease = None
        try:
            if cancel_active:
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                first_lease.release()
            else:
                second_lease = runtime.reserve()
                second = asyncio.create_task(runtime.run(second_lease, work))
                await until(lambda: runtime.waiting == 1)
                second.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await second
                second_lease.release()
            assert runtime.admitted == runtime.active == 1
            release.set()
            if not cancel_active:
                assert await first
            await until(lambda: runtime.admitted == 0)
            assert runtime.active == runtime.waiting == 0
        finally:
            release.set()
            await runtime.close()

    asyncio.run(scenario())


def test_upload_timeout_and_oversize_release_capacity(monkeypatch):
    async def scenario():
        app, runtime = harness(lambda: {'ok': True})
        messages = []

        async def never_receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        try:
            scope = {'type': 'http', 'method': 'POST', 'path': '/api/predict', 'headers': []}
            await InferenceAdmission(UploadMemoryLimit(app), runtime)(scope, never_receive, send)
            assert messages[0]['status'] == 408 and runtime.admitted == 0
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await client.post('/api/predict', content=b'',
                                             headers={'content-length': str(PREDICT_BODY + 1)})
                assert response.status_code == 413 and runtime.admitted == 0
                assert (await client.post('/api/predict')).status_code == 200
        finally:
            await runtime.close()

    monkeypatch.setattr('upload_memory.UPLOAD_SECONDS', .02)
    asyncio.run(scenario())


def test_health_readiness_and_privacy_headers(client, monkeypatch):
    import app
    health = client.get('/api/health')
    assert health.status_code == 200 and health.json()['alive']
    assert health.headers['cache-control'] == 'no-store'
    assert client.get('/api/ready').json() == {'ready': True}
    service = client.get('/api/status').json()['service']
    assert service['policy'] == 'bounded_inference_v1' and service['capacity'] == 3
    assert service['active'] == service['admitted'] == 0
    monkeypatch.setattr(app.engine, 'cnn_model', None)
    assert client.get('/api/ready').status_code == 503
    assert client.get('/api/health').json()['alive']
