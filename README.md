# YAWN - Yet Another Whisper traNscriber

Just a small afternoon project using [PyAudio](https://pypi.org/project/PyAudio/), [whisper](https://github.com/openai/whisper) and [keyboard](https://github.com/boppreh/keyboard) in order to transcribe speech at the press of a button.

## Use
```yaml
usage: yawn.EXE [-h] [-k HOTKEY] [-m MODEL]

Yet Another Whisper Transcriber Records your voice while the hotkey is pressed, transcribes it, then writes it to whatever window you have open.

options:
  -h, --help            show this help message and exit
  -k HOTKEY, --hotkey HOTKEY
                        The hotkey to start/stop recording (defaults to "scroll lock")
  -m MODEL, --model MODEL
                        The whisper model to be used (defaults to "base.en")
```


1. Keep the hotkey (`scroll lock` by default) pressed to record your voice.
2. Let go to let Whisper transcribe it.
3. The output is written to whatever application you have open at the time.
