from __future__ import annotations

import base64

from server import create_app


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_observation_and_prompt_roundtrip() -> None:
    client = create_app().test_client()
    response = client.post(
        "/api/observation",
        json={"base_image": base64.b64encode(PNG_1X1).decode(), "metadata": {"frame_id": 7}},
    )
    assert response.status_code == 202
    assert client.get("/api/state").json["observation"]["frame_id"] == 7
    assert client.get("/api/observation/base").data == PNG_1X1

    response = client.post(
        "/api/prompt",
        json={
            "mode": "combined",
            "prompt_image": "data:image/png;base64," + base64.b64encode(PNG_1X1).decode(),
            "draw_ops": [{"type": "point", "point_hw": [0, 0]}],
            "prompt_2d_drag": [0.25, -0.5],
            "prompt_global_motion": [0.1, 0.0, -0.2],
            "phase2_steps": 2.4,
        },
    )
    assert response.status_code == 201
    prompt = client.get("/api/prompt?after=0").json
    assert prompt["prompt_2d_drag_mask"] is True
    assert prompt["prompt_global_motion_mask"] is True
    assert prompt["sample_kwargs"]["phase2_steps"] == 2.4
    assert client.get(f"/api/prompt?after={prompt['sequence']}").status_code == 204
    ack = client.post(f"/api/prompt/{prompt['sequence']}/ack")
    assert ack.status_code == 200
    assert client.get("/api/state").json["acknowledged_prompt_sequence"] == prompt["sequence"]


def test_validation() -> None:
    client = create_app().test_client()
    assert client.post("/api/observation", json={}).status_code == 400
    assert client.post("/api/prompt", json={"prompt_global_motion": [2, 0, 0]}).status_code == 400


def test_pages_cross_origin_roundtrip() -> None:
    origin = "https://robot-demo.github.io"
    client = create_app(allowed_origins=[origin]).test_client()
    headers = {"Origin": origin}
    preflight = client.options("/api/prompt", headers={
        **headers,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
        "Access-Control-Request-Private-Network": "true",
    })
    assert preflight.status_code == 200
    assert preflight.headers["Access-Control-Allow-Origin"] == origin
    assert "POST" in preflight.headers["Access-Control-Allow-Methods"]
    assert preflight.headers["Access-Control-Allow-Headers"] == "Content-Type"
    assert preflight.headers["Access-Control-Allow-Private-Network"] == "true"
    # Python client has no Origin; browser fetches camera bytes with CORS.
    assert client.post("/api/observation", json={
        "base_image": base64.b64encode(PNG_1X1).decode(),
    }).status_code == 202
    for path in ("/api/state", "/api/observation/base"):
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert response.headers["Access-Control-Allow-Origin"] == origin
        assert "Origin" in response.headers["Vary"]
    submitted = client.post("/api/prompt", json={"prompt_global_motion": [0.2, 0, 0]}, headers=headers)
    assert submitted.status_code == 201
    assert submitted.headers["Access-Control-Allow-Origin"] == origin
    prompt = client.get("/api/prompt?after=0").json
    assert prompt["prompt_global_motion"] == [0.2, 0, 0]
    assert client.post(f"/api/prompt/{prompt['sequence']}/ack").status_code == 200
    assert client.get("/api/state", headers=headers).json["acknowledged_prompt_sequence"] == prompt["sequence"]


def test_disallowed_origin_cannot_mutate_state() -> None:
    client = create_app(allowed_origins=["https://robot-demo.github.io"]).test_client()
    headers = {"Origin": "https://untrusted.example"}
    for method in ("OPTIONS", "POST", "DELETE"):
        response = client.open("/api/prompt", method=method, headers=headers, json={})
        assert response.status_code == 403
        assert "Access-Control-Allow-Origin" not in response.headers
    assert client.get("/api/state").json["prompt_sequence"] == 0
    assert client.post("/api/prompt", json={}, headers={"Origin": "http://localhost"}).status_code == 201
