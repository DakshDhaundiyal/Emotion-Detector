"""Emotion detector with HTTP 400 handling."""
import requests

EMOTION_URL = "https://sn-watson-nlp-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"

def emotion_detector(text_to_analyze):
    headers = {"grpc-metadata-mm-model-id": MODEL_ID}
    response = requests.post(
        EMOTION_URL,
        headers=headers,
        json={"raw_document": {"text": text_to_analyze}},
        timeout=30
    )
    if response.status_code == 400:
        return {
            "anger": None, "disgust": None, "fear": None,
            "joy": None, "sadness": None, "dominant_emotion": None
        }
    response.raise_for_status()
    return response.text
