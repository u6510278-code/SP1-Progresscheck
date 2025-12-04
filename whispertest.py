import whisper

def main():
    print("Loading model...")
    model = whisper.load_model("tiny")
    print("Model loaded.")

    audio_path = "testing.mp3"

    print(f"Transcribing {audio_path} ...")
    result = model.transcribe(audio_path)

    print("=== TRANSCRIPTION ===")
    print(result["text"])

if __name__ == "__main__":
    main()
