import argparse
import os
import threading
import time

import keyboard
import numpy as np
import pyaudio
import torch
import whisper
from halo import Halo

CHUNKSIZE = 1024
SAMPLING_RATE = 16000


class Recorder:
    def __init__(self) -> None:
        self.recording_thread: threading.Thread = None
        self.stop_requested = threading.Event()
        self.frames: list[np.ndarray] = []
        self.recording_spinner = Halo(text="Recording", spinner='dots')

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
            self.recording_spinner.start()
            self.recording_thread = threading.Thread(target=self._record)
            self.recording_thread.start()

    def stop_recording(self):
        self.stop_requested.set()
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
            self.stop_requested.clear()
        self.recording_thread = None
        self.recording_spinner.stop()


@Halo("Processing audio", spinner="dots")
def pre_process_audio(frames):
    audio_data = np.concatenate(frames) # Consolidate list of arrays
    audio_data = np.frombuffer(audio_data, np.int16).flatten().astype(np.float32) / 32768.0 # type conversion magic
    return audio_data


def transcribe(model: whisper.Whisper, audio: np.ndarray) -> str:
    progress = Halo("Transcribing", spinner="dots").start()

    # pad/trim audio to match expected model input
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode
    res = model.transcribe(audio, fp16=False, language="en")

    progress.stop()

    text = res["text"].strip()
    lang = res["language"]
    print(f"[{lang}]: {text}")
    return text


def parse_command_arguments() -> tuple[str, str]:
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
        help='The hotkey to start/stop recording (defaults to "scroll lock")',
    )
    parser.add_argument(
        "-m", '--model',
        type=str,
        default='base.en',
        help='The whisper model to be used (defaults to "base.en")',
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=0,
        help='Nr. of threads used by torch in case of CPU inference'
    )

    args, unknown_args = parser.parse_known_args()

    return args, unknown_args


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
    recorder = Recorder()
    status = Halo(spinner="dots")

    # CLI handling
    args, unknown_args = parse_command_arguments()
    if len(unknown_args) > 0:
        raise ValueError(f"Unknown arguments: {', '.join(unknown_args)}")
    if (threads := args.threads) > 0:
        torch.set_num_threads(threads)

    if torch.cuda.is_available():
        status.succeed("Cuda enabled.")
    else:
        status.fail("Cuda not availabe, using CPU instead.")

    status.start("Loading model")
    model = whisper.load_model(args.model)
    status.succeed("Model loaded.")

    keyboard.add_hotkey(args.hotkey, record, args=[recorder], suppress=True, trigger_on_release=False)
    keyboard.add_hotkey(args.hotkey, stop_recording, args=[model, recorder], suppress=True, trigger_on_release=True)

    status.succeed("Ready!")
    status.info(f"Press and hold `{args.hotkey}` to start recording.")

    while True:
        time.sleep(10)
