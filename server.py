import asyncio
from typing import Tuple
import threading
import time

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Request
from faster_whisper import WhisperModel
import subprocess



app = FastAPI()

MODEL_TYPE = "nbv/nbvnbvATCmodelv2"
RUN_TYPE = "gpu"  # "cpu" or "gpu"

# For CPU usage (https://github.com/SYSTRAN/faster-whisper/issues/100#issuecomment-1492141352)
NUM_WORKERS = 10
CPU_THREADS = 4

# For GPU usage
GPU_DEVICE_INDICES = [0, 1]

VAD_FILTER = True


def create_whisper_model() -> WhisperModel:
    if RUN_TYPE.lower() == "gpu":
        whisper = WhisperModel(MODEL_TYPE,
                               device="cuda",
                               compute_type="float16",
                               device_index=GPU_DEVICE_INDICES,
                               download_root="./models")
    elif RUN_TYPE.lower() == "cpu":
        whisper = WhisperModel(MODEL_TYPE,
                               device="cpu",
                               compute_type="int8",
                               num_workers=NUM_WORKERS,
                               cpu_threads=CPU_THREADS,
                               download_root="./models")
    else:
        raise ValueError(f"Invalid model type: {RUN_TYPE}")

    print("Loaded model")

    return whisper


model = create_whisper_model()
print("Loaded model")


async def parse_body(request: Request):
    data: bytes = await request.body()
    return data

def convert_number_words_to_digits(sentence):
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    }

    words = sentence.split()
    for i, word in enumerate(words):
        if word.lower() in number_words:
            words[i] = number_words[word.lower()]

    return ' '.join(words)



def execute_blocking_whisper_prediction(
        model: WhisperModel,
        audio_data_array: np.ndarray,
        language_code: str = "") -> Tuple[str, str, float]:
    language_code = language_code.lower().strip()
    segments, info = model.transcribe(
        audio_data_array,
        language=language_code if language_code != "" else None,
        beam_size=6, #default 5
        vad_filter=VAD_FILTER,
        vad_parameters=dict(min_silence_duration_ms=500)) #default 500
    segments = [s.text for s in segments]
    transcription = " ".join(segments)
    transcription = transcription.strip()
    transcription = convert_number_words_to_digits(transcription)
    return transcription, info.language, info.language_probability


@app.post("/predict")
async def predict(
        audio_data: bytes = Depends(parse_body), language_code: str = ""):
    # Convert the audio bytes to a NumPy array
    audio_data_array: np.ndarray = np.frombuffer(audio_data, np.int16).astype(
        np.float32) / 255.0

    try:
        # Run the prediction on the audio data
        result = await asyncio.get_running_loop().run_in_executor(
            None, execute_blocking_whisper_prediction, model, audio_data_array,
            language_code)

    except Exception as e:
        print(e)
        result = "Error"

    return {
        "prediction": result[0],
        "language": result[1],
        "language_probability": result[2]
    }


def run_localtunnel():
    # Start the LocalTunnel process and capture its output

    ssh_command = f"!ssh -R 80:127.0.0.1:8008 serveo.net -o StrictHostKeyChecking=no"

    lt_process = subprocess.Popen(["ssh", "-R", "nbvnbv.serveo.net:80:127.0.0.1:8008", "serveo.net", "-o", "StrictHostKeyChecking=no"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Parse the output to get the URL
    for line in lt_process.stdout:
        if "your url is:" in line:
            print(line.strip())
            break
    
    return lt_process


if __name__ == "__main__":

    lt_thread = threading.Thread(target=run_localtunnel)
    lt_thread.start()

    time.sleep(2)

    #from pyngrok import ngrok
    #ngrok.set_auth_token("2CyddSn0XrK93yRlk0n3K3moVLi_5uk1JDY9aSt5voT4koC4T")
    #ngrok_tunnel2 = ngrok.connect("8008")
    #print(ngrok_tunnel2.public_url)

    uvicorn.run(app, host="127.0.0.1", port=8008)

