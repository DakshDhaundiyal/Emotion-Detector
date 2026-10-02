"""Emotion detection application using the Watson NLP library."""

import requests


def emotion_detector(text_to_analyze):
    """Detect emotions in the supplied text using Watson NLP."""
    url = (
        "https://sn-watson-nlp-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(url, json=payload, timeout=30)

    if response.status_code == 200:
        result = response.json()
        emotions = result["emotionPredictions"][0]["emotion"]

        anger = emotions.get("anger")
        disgust = emotions.get("disgust")
        fear = emotions.get("fear")
        joy = emotions.get("joy")
        sadness = emotions.get("sadness")

        dominant_emotion = max(emotions, key=emotions.get)

        return {
            "anger": anger,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness,
            "dominant_emotion": dominant_emotion,
        }

    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }
