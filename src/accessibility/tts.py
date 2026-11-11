"""
tts.py
------
Text-to-Speech engine for Lumen Reader.
Primary  : pyttsx3 (offline, no API needed)
Fallback : gTTS   (online, better voice quality)
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Literal


ENGINE = Literal["pyttsx3", "gtts"]


class TTSEngine:
    """
    Unified TTS interface.
    Tries pyttsx3 first (offline). Falls back to gTTS if needed.
    """

    def __init__(self, engine: ENGINE = "pyttsx3", rate: int = 150, volume: float = 1.0):
        """
        Args:
            engine : 'pyttsx3' (offline) or 'gtts' (online)
            rate   : Speech rate — words per minute (pyttsx3 only)
            volume : Volume 0.0 to 1.0 (pyttsx3 only)
        """
        self.engine_name = engine
        self.rate        = rate
        self.volume      = volume
        self._engine     = None

        if engine == "pyttsx3":
            self._init_pyttsx3()

    # ── Public API ─────────────────────────────────────────────────

    def speak(self, text: str):
        """
        Speak text aloud immediately (blocking).

        Args:
            text: Plain text string to speak.
        """
        if not text or not text.strip():
            return

        clean = self._clean_for_speech(text)

        if self.engine_name == "pyttsx3" and self._engine:
            self._speak_pyttsx3(clean)
        else:
            self._speak_gtts(clean)

    def save_audio(self, text: str, output_path: str) -> str:
        """
        Save spoken text to an MP3 file.

        Args:
            text        : Text to convert
            output_path : Where to save the .mp3

        Returns:
            Path to saved file.
        """
        clean = self._clean_for_speech(text)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.engine_name == "pyttsx3" and self._engine:
            self._engine.save_to_file(clean, output_path)
            self._engine.runAndWait()
        else:
            from gtts import gTTS
            tts = gTTS(text=clean, lang="en", slow=False)
            tts.save(output_path)

        print(f"[TTS] Saved audio → {output_path}")
        return output_path

    def speak_answer(self, answer: str, sources: list):
        """
        Speak a RAG answer with its source citations.
        Formats citations naturally for audio output.

        Args:
            answer  : Answer text from the RAG graph
            sources : Source list from run_query()
        """
        # Speak the answer
        self.speak(answer)

        # Speak citations if any
        if sources:
            time.sleep(0.4)
            citation_parts = []
            for i, s in enumerate(sources[:3], 1):
                page  = s.get("page", "unknown")
                fname = s.get("filename", "document")
                label = s.get("label", "")
                citation_parts.append(
                    f"Source {i}: {fname}, page {page}, topic {label}."
                )
            citation_text = "This answer was found in: " + " ".join(citation_parts)
            self.speak(citation_text)

    def set_rate(self, rate: int):
        """Change speech rate (pyttsx3 only)."""
        self.rate = rate
        if self._engine:
            self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        """Change volume 0.0–1.0 (pyttsx3 only)."""
        self.volume = volume
        if self._engine:
            self._engine.setProperty("volume", volume)

    def list_voices(self) -> list:
        """Return available voices (pyttsx3 only)."""
        if not self._engine:
            return []
        voices = self._engine.getProperty("voices")
        return [{"id": v.id, "name": v.name, "lang": v.languages} for v in voices]

    def set_voice(self, voice_index: int = 0):
        """Set voice by index from list_voices() (pyttsx3 only)."""
        if not self._engine:
            return
        voices = self._engine.getProperty("voices")
        if voice_index < len(voices):
            self._engine.setProperty("voice", voices[voice_index].id)
            print(f"[TTS] Voice set to: {voices[voice_index].name}")

    # ── Private ────────────────────────────────────────────────────

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   self.rate)
            self._engine.setProperty("volume", self.volume)
            print(f"[TTS] pyttsx3 initialised (rate={self.rate}, volume={self.volume})")
        except Exception as e:
            print(f"[TTS] pyttsx3 failed ({e}), falling back to gTTS")
            self.engine_name = "gtts"
            self._engine     = None

    def _speak_pyttsx3(self, text: str):
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f"[TTS] pyttsx3 speak failed ({e}), trying gTTS")
            self._speak_gtts(text)

    def _speak_gtts(self, text: str):
        try:
            from gtts import gTTS
            import pygame
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            gTTS(text=text, lang="en", slow=False).save(tmp_path)
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.quit()
            os.unlink(tmp_path)
        except ImportError:
            print("[TTS] pygame not installed. Install: pip install pygame")
            print(f"[TTS] Would have spoken: {text[:100]}...")
        except Exception as e:
            print(f"[TTS] gTTS speak failed ({e})")
            print(f"[TTS] Text: {text[:100]}...")

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """
        Strip markdown and symbols that sound bad when spoken aloud.
        Converts markdown to natural spoken equivalents.
        """
        import re
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # bold
        text = re.sub(r"\*(.*?)\*",     r"\1", text)   # italic
        text = re.sub(r"`(.*?)`",       r"\1", text)   # inline code
        text = re.sub(r"#+\s*",         "",    text)   # headers
        text = re.sub(r"\[.*?\]\(.*?\)","",    text)   # links
        text = re.sub(r"\|",            " ",   text)   # table pipes
        text = re.sub(r"-{2,}",         " ",   text)   # horizontal rules
        text = re.sub(r"\n{2,}",        ". ",  text)   # paragraph breaks → pause
        text = re.sub(r"\s+",           " ",   text)   # collapse whitespace
        return text.strip()


# ── Quick test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing TTS engine...")
    tts = TTSEngine(engine="pyttsx3", rate=150)

    print("\nAvailable voices:")
    for i, v in enumerate(tts.list_voices()):
        print(f"  [{i}] {v['name']}")

    tts.speak("Hello. I am Lumen Reader, your accessible document assistant.")
    tts.speak("Phase 5 text to speech is working correctly.")

    tts.save_audio("This is a saved audio test.", "data/tts_test.mp3")
    print("\nTTS test complete.")