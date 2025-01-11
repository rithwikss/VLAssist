import pyttsx3
import threading

class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.7)  # Volume (0.0 to 1.0)
        self.lock = threading.Lock()

    def speak(self, text):
        def run_speak():
            with self.lock:
                self.engine.say(text)
                self.engine.runAndWait()

        thread = threading.Thread(target=run_speak)
        thread.start()

tts_engine = TTSEngine()

def speak_text(text):
    tts_engine.speak(text)