from gtts import gTTS
import os
import uuid

def text_to_speech(text, lang='en'):
    filename = f"response_{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang=lang)
    path = os.path.join("static", filename)
    tts.save(path)
    return path
