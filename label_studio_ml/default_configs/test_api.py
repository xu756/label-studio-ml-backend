"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""

import pytest
import json
from model import NewModel


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=NewModel)
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                # Your input test data here
            }
        }],
        # Your labeling configuration here
        'label_config': '<View></View>'
    }

    expected_response = {
        'results': [{
            # Your expected result here
        }]
    }

    response = client.post('/predict', json=request)
    assert response.status_code == 200
    assert response.json() == expected_response

