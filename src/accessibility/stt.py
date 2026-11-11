"""
stt.py
------
Speech-to-Text engine for Lumen Reader using OpenAI Whisper.
Runs fully offline on CPU. No API key needed.
"""

import os
import wave
import tempfile
from pathlib import Path
from typing import Optional


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# Model sizes: tiny, base, small, medium, large
# base = good balance of speed + accuracy for English
# tiny = fastest, lower accuracy
# small = better accuracy, slower


class STTEngine:
    """
    Whisper-based speech-to-text.
    Records from microphone or transcribes an audio file.
    """

    def __init__(self, model_size: str = WHISPER_MODEL):
        """
        Args:
            model_size: Whisper model — 'tiny', 'base', 'small', 'medium'
        """
        self.model_size = model_size
        self._model     = None
        print(f"[STT] Whisper model: {model_size} (loads on first use)")

    # ── Public API ─────────────────────────────────────────────────

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to .wav, .mp3, or .m4a file

        Returns:
            Transcribed text string.
        """
        self._ensure_model()
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"[STT] Transcribing: {audio_path}")
        result = self._model.transcribe(audio_path, language="en", fp16=False)
        text   = result["text"].strip()
        print(f"[STT] Transcribed: {text[:80]}...")
        return text

    def record_and_transcribe(
        self,
        duration: int = 5,
        sample_rate: int = 16000,
    ) -> str:
        """
        Record from microphone and transcribe immediately.

        Args:
            duration    : Recording duration in seconds
            sample_rate : Audio sample rate (16000 recommended for Whisper)

        Returns:
            Transcribed text string.
        """
        self._ensure_model()
        audio_path = self._record_audio(duration, sample_rate)

        if not audio_path:
            return ""

        text = self.transcribe_file(audio_path)

        # Clean up temp file
        try:
            os.unlink(audio_path)
        except Exception:
            pass

        return text

    def listen_for_query(self, prompt: str = "Listening...") -> str:
        """
        High-level method: show prompt, record 6 seconds, return transcript.
        Used directly by the Streamlit UI in Phase 6.

        Args:
            prompt: Message to display while recording

        Returns:
            User's spoken question as text.
        """
        print(f"[STT] {prompt}")
        text = self.record_and_transcribe(duration=6)
        if text:
            print(f"[STT] You said: {text}")
        else:
            print("[STT] No speech detected.")
        return text

    # ── Private ────────────────────────────────────────────────────

    def _ensure_model(self):
        """Lazy-load Whisper model on first use."""
        if self._model is None:
            try:
                import whisper
                print(f"[STT] Loading Whisper '{self.model_size}' model...")
                self._model = whisper.load_model(self.model_size)
                print(f"[STT] Whisper ready.")
            except ImportError:
                raise ImportError(
                    "openai-whisper not installed. Run: pip install openai-whisper"
                )

    def _record_audio(
        self,
        duration: int,
        sample_rate: int,
    ) -> Optional[str]:
        """
        Record audio from default microphone.
        Returns path to temp .wav file, or None on failure.
        """
        try:
            import sounddevice as sd
            import numpy as np

            print(f"[STT] Recording for {duration}s... speak now")
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()  # block until done
            print("[STT] Recording complete.")

            # Save to temp .wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())

            return tmp_path

        except ImportError:
            print("[STT] sounddevice not installed. Run: pip install sounddevice")
            return None
        except Exception as e:
            print(f"[STT] Recording failed: {e}")
            return None


# ── Quick test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    stt = STTEngine(model_size="base")

    print("\nSTT Test Options:")
    print("  1. Record from mic (5 seconds)")
    print("  2. Transcribe existing audio file")
    choice = input("Choose [1/2]: ").strip()

    if choice == "1":
        try:
            import sounddevice
            text = stt.record_and_transcribe(duration=5)
            print(f"\nTranscribed: {text}")
        except ImportError:
            print("Install sounddevice first: pip install sounddevice")

    elif choice == "2":
        path = input("Enter audio file path: ").strip()
        text = stt.transcribe_file(path)
        print(f"\nTranscribed: {text}")

    print("\nSTT test complete.")