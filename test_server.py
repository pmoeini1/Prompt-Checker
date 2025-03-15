import pytest
from server import app  # Import your Flask app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_getFN(client):
    response = client.get('/getFN')
    assert response.status_code == 200
    assert len(response.get_data(as_text=True)) > 0


def test_getFP(client):
    response = client.get('/getFP')
    assert response.status_code == 200
    assert len(response.get_data(as_text=True)) > 0
