import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope='session')
def client():
    from app import app
    from fastapi.testclient import TestClient
    with TestClient(app) as value:
        assert value.get('/api/status').json()['ready']
        yield value
