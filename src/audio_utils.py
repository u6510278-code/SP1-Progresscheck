
import os
import whisper


DEFAULT_WHISPER_MODEL = "tiny"

_model_cache = None

def _get_model(model_name: str = DEFAULT_WHISPER_MODEL):
    global _model_cache
    if _model_cache is None:
        print(f"[INFO] Loading Whisper model: {model_name}")
        _model_cache = whisper.load_model(model_name)
    return _model_cache


def transcribe_audio(path: str, model_name: str = DEFAULT_WHISPER_MODEL) -> str:
    """
    Transcribe an audio file to plain text using local Whisper.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    model = _get_model(model_name)

    print(f"[INFO] Transcribing audio: {path}")
    result = model.transcribe(path)

    text = result.get("text", "") or ""
    print(f"[INFO] Transcription length: {len(text)} characters")
    return text.strip()
