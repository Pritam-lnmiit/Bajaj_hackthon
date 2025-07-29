import speech_recognition as sr
import requests

recognizer = sr.Recognizer()

print("🎙 Speak now...")

with sr.Microphone() as source:
    audio = recognizer.listen(source)

try:
    query = recognizer.recognize_google(audio)
    print("🗣 You said:", query)

    response = requests.post(
        "http://127.0.0.1:8000/voice-query",
        json={"text": query}
    )

    print("🤖 API Response:", response.json())

except sr.UnknownValueError:
    print("❌ Could not understand the audio")
except sr.RequestError as e:
    print(f"❌ Error with Google Speech Recognition: {e}")
