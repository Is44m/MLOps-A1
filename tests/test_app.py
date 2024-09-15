import pytest
import json
from api.app import app  # Ensure this import is correct

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_valid_input(client):
    response = client.post('/predict', json={
        'area': 7420,
        'bedrooms': 4,
        'bathrooms': 2,
        'stories': 3,
        'mainroad': 'yes',
        'guestroom': 'no',
        'basement': 'no',
        'hotwaterheating': 'no',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'furnished'
    })
    assert response.status_code == 200
    assert 'predicted_price' in response.json

def test_invalid_bathrooms(client):
    response = client.post('/predict', json={
        'area': 7420,
        'bedrooms': 4,
        'bathrooms': -2,
        'stories': 3,
        'mainroad': 'yes',
        'guestroom': 'no',
        'basement': 'no',
        'hotwaterheating': 'no',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'furnished'
    })
    assert response.status_code == 400
    assert 'error' in response.json

def test_invalid_furnishingstatus(client):
    response = client.post('/predict', json={
        'area': 7420,
        'bedrooms': 4,
        'bathrooms': 2,
        'stories': 3,
        'mainroad': 'yes',
        'guestroom': 'no',
        'basement': 'no',
        'hotwaterheating': 'no',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'luxury'
    })
    assert response.status_code == 400
    assert 'error' in response.json
