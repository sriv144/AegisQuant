import importlib

from fastapi.testclient import TestClient


def test_dashboard_api_requires_key(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGIS_API_KEY", "test-key")
    monkeypatch.setenv("POSTGRES_URL", f"sqlite:///{tmp_path / 'auth.db'}")

    import src.webapp.server as server

    server = importlib.reload(server)
    client = TestClient(server.app)

    unauthenticated = client.get("/api/portfolio")
    assert unauthenticated.status_code == 401

    authenticated = client.get("/api/portfolio", headers={"Authorization": "Bearer test-key"})
    assert authenticated.status_code == 200
    assert "history" in authenticated.json()
