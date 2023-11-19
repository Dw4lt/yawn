import argparse
import threading
import time

import keyboard
import numpy as np
import pyaudio
import torch
import whisper

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


def handle_command_arguments():
    parser = argparse.ArgumentParser(
        description="""
        Yet Another Whisper Transcriber.
        Records your voice while the hotkey is pressed, transcribes it, then writes it to whatever window you have open.
        """
    )

    parser.add_argument(
        "-k", '--hotkey',
        type=str,
        default='scroll lock',
        help='The hotkey to start/stop recording (default: "scroll lock")',
    )

    return parser.parse_args()


def record(recorder: Recorder):
    recorder.start_recording()


def stop_recording(model: whisper.Whisper, recorder: Recorder):
    recorder.stop_recording()
    if len(recorder.frames) > 0:
        audio = pre_process_audio(recorder.frames)

        text = transcribe(model, audio)
        keyboard.write(text)
    else:
        print("No audio recorded.")


def main():
    args = handle_command_arguments()

    print(f"Cuda: {torch.cuda.is_available()}")

    recorder = Recorder()
    model = whisper.load_model("base.en")


    keyboard.add_hotkey(args.hotkey, record, args=[recorder], suppress=True, trigger_on_release=False)
    keyboard.add_hotkey(args.hotkey, stop_recording, args=[model, recorder], suppress=True, trigger_on_release=True)

    print("ready!")
    while True:
        time.sleep(10)

