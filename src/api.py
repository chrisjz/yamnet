import csv
import io
import os
from typing import Any

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy
from scipy.io import wavfile
from pydub import AudioSegment

# Initialize FastAPI app
app = FastAPI()

# Load the YAMNet model
model = hub.load("https://tfhub.dev/google/yamnet/1")


def class_names_from_csv(class_map_csv_text: str) -> list[str]:
    """
    Load class names from a CSV file.

    Args:
        class_map_csv_text (str): Path to the CSV file containing class names.

    Returns:
        list[str]: List of class names.
    """
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])
    return class_names


class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)


def ensure_sample_rate(
    original_sample_rate: int, waveform: np.ndarray, desired_sample_rate: int = 16000
) -> tuple[int, np.ndarray]:
    """
    Resample waveform if required.

    Args:
        original_sample_rate (int): The original sample rate of the waveform.
        waveform (np.ndarray): The waveform data.
        desired_sample_rate (int, optional): The desired sample rate. Defaults to 16000.

    Returns:
        tuple[int, np.ndarray]: Tuple containing the resampled rate and waveform.
    """
    if original_sample_rate != desired_sample_rate:
        desired_length = int(
            round(float(len(waveform)) / original_sample_rate * desired_sample_rate)
        )
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def convert_mp3_to_wav(file: UploadFile) -> io.BytesIO:
    """
    Convert an MP3 file to WAV format.

    Args:
        file (UploadFile): The MP3 file to convert.

    Returns:
        io.BytesIO: The WAV file as a BytesIO object.
    """
    audio = AudioSegment.from_file(io.BytesIO(file.file.read()), format="mp3")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io


def inference(wav_data: io.BytesIO, top_n: int) -> list[dict[str, Any]]:
    """
    Perform inference on the given WAV data.

    Args:
        wav_data (io.BytesIO): The WAV data.
        top_n (int): The number of top results to return.

    Returns:
        list[dict[str, Any]: A list of dictionaries containing class names and scores.
    """
    sample_rate, wav_data = wavfile.read(wav_data)
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    if wav_data.ndim >= 2:
        wav_data = wav_data[:, 0]

    waveform = wav_data / tf.int16.max

    # Run the model
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy().mean(axis=0)
    top_indices = np.argsort(scores_np)[-top_n:][::-1]
    results = [
        {"class": class_names[i], "score": round(float(scores_np[i]), 4)}
        for i in top_indices
    ]
    return results


@app.post("/classify")
async def classify_audio(file: UploadFile, top_n: int = 5) -> JSONResponse:
    """
    Classify an uploaded audio file.

    Args:
        file (UploadFile): The audio file to classify.
        top_n (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        JSONResponse: A JSON response containing the classification results.
    """
    filename = file.filename
    extension = os.path.splitext(filename)[1].lower()

    if extension not in [".wav", ".mp3"]:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only WAV and MP3 are supported."
        )

    # Convert MP3 to WAV if necessary
    if extension == ".mp3":
        wav_io = convert_mp3_to_wav(file)
    else:
        wav_io = io.BytesIO(file.file.read())

    try:
        results = inference(wav_io, top_n)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
