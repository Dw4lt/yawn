import whisper

import keyboard

import pyaudio
import numpy as np
import threading

CHUNKSIZE = 1024
SAMPLING_RATE = 16000


class Recorder:
    def __init__(self) -> None:
        self.recording_thread: threading.Thread = None
        self.stop_requested = threading.Event()
        self.frames: list[np.ndarray] = []

    def _record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=CHUNKSIZE)
        self.frames = []

        while not self.stop_requested.is_set():
            data = stream.read(CHUNKSIZE, exception_on_overflow=False)
            numpydata = np.frombuffer(data, dtype=np.int16)
            self.frames.append(numpydata)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def start_recording(self):
        if self.recording_thread is None:
            print("recording on")
            self.recording_thread = threading.Thread(target=self._record)
            self.recording_thread.start()

    def stop_recording(self):
        print("stopping recording")
        self.stop_requested.set()
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
            self.stop_requested.clear()
        self.recording_thread = None


def pre_process_audio(frames):
    audio_data = np.concatenate(frames) # Consolidate list of arrays
    audio_data = np.frombuffer(audio_data, np.int16).flatten().astype(np.float32) / 32768.0 # type conversion magic
    return audio_data


def transcribe(model: whisper.Whisper, audio: np.ndarray) -> str:
    print("transcribing")
    # pad/trim audio to match expected model input
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode
    res = model.transcribe(audio, fp16=False, language="en")

    text = res["text"].strip()
    lang = res["language"]
    print(f"[{lang}]: {text}")
    return text


def main():
    r = Recorder()
    model = whisper.load_model("base.en")

    def record():
        r.start_recording()

    def stop_recording():
        r.stop_recording()
        if len(r.frames) > 0:
            audio = pre_process_audio(r.frames)

            text = transcribe(model, audio)
            keyboard.write(text)
        else:
            print("No audio recorded.")


    keyboard.add_hotkey("scroll lock", record, suppress=True, trigger_on_release=False)
    keyboard.add_hotkey("scroll lock", stop_recording, suppress=True, trigger_on_release=True)

    import torch
    print(f"Cuda: {torch.cuda.is_available()}")

    # Tkinter setup
    import time
    print("ready!")
    while True:
        time.sleep(10)

