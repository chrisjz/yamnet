# YAMNet Audio Event Classification API

This project provides a FastAPI-based web API that uses the YAMNet model to
classify audio events. The API takes an audio file (WAV or MP3) as input and
returns a list of the top classifications along with their associated
confidence scores.

## Features

- **Audio Classification**: Classify audio events using the YAMNet model.
- **Supports Multiple Formats**: Accepts both WAV and MP3 files.
- **GPU Acceleration**: Utilizes GPU if available for faster inference.

## Installation

### Prerequisites

- **Docker**: Make sure Docker is installed and running on your machine.
- **NVIDIA Docker**: If you want to use GPU, ensure that the NVIDIA
Docker Toolkit is installed.

### Clone the Repository

```bash
git clone https://github.com/chrisjz/yamnet.git
cd yamnet
```

### Build and Run the Docker Container

Use Docker Compose to build the Docker image and start the container:

```bash
docker-compose up --build
```

This will build the image and run the FastAPI application. The API will
be available at `http://localhost:8000`.

## API Usage

### Endpoint: `/classify`

- **Method**: `POST`
- **URL**: `http://localhost:8000/classify`
- **Query Parameters**:
  - `top_n` (optional): Number of top results to return. Default is `5`.
- **Body**: Multipart form-data
  - `file`: The audio file (WAV or MP3) to be classified.

### Example Request

#### Using CURL

```bash
curl -X POST "http://localhost:8000/classify?top_n=3" \
-F "file=@path_to_your_audio_file.wav"
```

#### Using PowerShell

```powershell
$audioFilePath = "C:\path\to\your\audiofile.wav"
$uri = "http://localhost:8000/classify?top_n=3"

$response = Invoke-RestMethod -Uri $uri -Method Post -InFile $audioFilePath -ContentType "multipart/form-data"

$response | ConvertTo-Json
```

#### Using Python

```python
import requests

url = "http://localhost:8000/classify"
audio_file_path = "C:\\path\\to\\your\\audiofile.wav"
params = {'top_n': 3}

with open(audio_file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files, params=params)

print(response.json())
```

#### Using Postman

1. Open Postman and create a new `POST` request.
2. Set the URL to `http://localhost:8000/classify`.
3. Add a query parameter `top_n` and set its value to `3`.
4. In the "Body" tab, select "form-data".
5. Add a file input with the key `file` and select your audio file.
6. Send the request.

### Example Response

```json
[
    {"class": "Animal", "score": 0.7132},
    {"class": "Domestic animals, pets", "score": 0.5123},
    {"class": "Cat", "score": 0.4236}
]
```

## Development

### Running Locally

If you prefer to run the API locally without Docker, follow these steps:

1. **Install Dependencies**: Install Poetry and use it to install dependencies.

   ```bash
   poetry install
   ```

2. **Run the Application**:

   ```bash
   poetry run uvicorn app:app --host 0.0.0.0 --port 8000
   ```

### Project Structure

- `app/api.py`: The main FastAPI application file.
- `Dockerfile`: Dockerfile for building the application image.
- `compose.yml`: Docker Compose configuration file.
- `pyproject.toml`: Poetry configuration file for managing dependencies.

## License

This project is licensed under the Apache 2.0 License.
See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for
any enhancements or bug fixes.

## Acknowledgments

- [YAMNet Model](https://www.kaggle.com/models/google/yamnet/tensorFlow2) by Google
- [thelou1s](https://github.com/thelou1s) - The [YAMNet inference script](https://huggingface.co/spaces/thelou1s/yamnet/blob/main/app.py)
this API is based on.
- [FastAPI](https://fastapi.tiangolo.com/) - The web framework used for this project.
- [TensorFlow](https://www.tensorflow.org/) - The deep learning framework
powering the model.
