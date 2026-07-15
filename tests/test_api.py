import pytest
from fastapi.testclient import TestClient
from label_studio_ml.api import init_app
from label_studio_ml.model import LabelStudioMLBase

@pytest.fixture
def client():
    app = init_app(model_class=LabelStudioMLBase)
    with TestClient(app) as client:
        yield client

def test_api(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'model_class': 'LabelStudioMLBase', 'status': 'UP'}

def test_metrics(client):
    response = client.get('/metrics')
    assert response.status_code == 200

def test_predict(client):
    response = client.post('/predict', json={
        'tasks': [{'id': 1}],
        'label_config': '<View></View>',
        'project': '1.1000000000',
        'params': {
            'context': {},
        },
    })
    assert response.status_code == 200

def test_setup(client):
    response = client.post('/setup', json={
        'project': '1.1000000000',
        'schema': '<View></View>',
        'extra_params': {}
    })
    assert response.status_code == 200

def test_webhook(client):
    response = client.post('/webhook', json={
        'action': 'ANNOTATION_CREATED',
        'project': {
            'id': 1,
            'label_config': '<View></View>'
        }
    })
    assert response.status_code == 201


