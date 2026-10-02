"""Formatted Watson NLP emotion detector."""
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

    if response.status_code != 200:
        response.raise_for_status()

    data = response.json()
    emotions = data["emotionPredictions"][0]["emotion"]

    result = {
        "anger": emotions.get("anger"),
        "disgust": emotions.get("disgust"),
        "fear": emotions.get("fear"),
        "joy": emotions.get("joy"),
        "sadness": emotions.get("sadness")
    }
    result["dominant_emotion"] = max(result, key=result.get)
    return result
