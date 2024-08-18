import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


@pytest.mark.parametrize(
    "audio_file,content_type,expected_status",
    [
        ("tests/data/miaow_16k.wav", "audio/wav", 200),
        ("tests/data/miaow_16k.mp3", "audio/mp3", 200),
        ("tests/data/miaow_16k.txt", "text/plain", 400),
    ],
)
def test_classify_audio(audio_file, content_type, expected_status):
    """
    Test the /classify endpoint with different file types.
    """
    with open(audio_file, "rb") as f:
        response = client.post(
            "/classify?top_n=3", files={"file": (audio_file, f, content_type)}
        )
    assert response.status_code == expected_status
    if response.status_code == 200:
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3
        for result in response.json():
            assert "class" in result
            assert "score" in result


def test_classify_audio_top_n():
    """
    Test the /classify endpoint with different top_n values.
    """
    audio_file = "tests/data/miaow_16k.wav"
    with open(audio_file, "rb") as f:
        response = client.post(
            "/classify?top_n=2", files={"file": (audio_file, f, "audio/wav")}
        )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 2


@pytest.mark.parametrize("top_n", [1, 3, 5])
def test_classify_audio_different_top_n(top_n):
    """
    Test the /classify endpoint with different top_n values.
    """
    audio_file = "tests/data/miaow_16k.wav"
    with open(audio_file, "rb") as f:
        response = client.post(
            f"/classify?top_n={top_n}", files={"file": (audio_file, f, "audio/wav")}
        )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == top_n


def test_classify_audio_invalid_file():
    """
    Test the /classify endpoint with an invalid file type.
    """
    audio_file = "tests/data/miaow_16k.txt"
    with open(audio_file, "rb") as f:
        response = client.post(
            "/classify", files={"file": (audio_file, f, "text/plain")}
        )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid file type. Only WAV and MP3 are supported."
    }
