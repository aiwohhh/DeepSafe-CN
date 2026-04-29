import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add api directory to path to import main
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api")))

from main import app, get_current_user

client = TestClient(app)


# Mock auth dependency to bypass login for some tests
async def mock_get_current_user():
    return {"username": "testuser", "disabled": False}


@pytest.fixture
def mock_auth():
    app.dependency_overrides[get_current_user] = mock_get_current_user
    yield
    app.dependency_overrides = {}


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "DeepSafe API"
    assert "configured_media_types" in data


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "overall_api_status" in data


def test_register_user():
    username = "pytest_user"
    response = client.post(
        "/register",
        data={"username": username, "password": "password123"},
    )
    assert response.status_code in [200, 400]


def test_login_user():
    # Register first
    client.post(
        "/register",
        data={"username": "login_test", "password": "password123"},
    )

    response = client.post(
        "/token", data={"username": "login_test", "password": "password123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_protected_route_without_auth():
    response = client.get("/users/me")
    assert response.status_code == 401


def test_protected_route_with_mock_auth(mock_auth):
    response = client.get("/users/me")
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
