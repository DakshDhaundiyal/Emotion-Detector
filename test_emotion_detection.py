"""Unit tests for the emotion detector."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Test emotion_detector output and error handling."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.01, "disgust": 0.01, "fear": 0.02,
                "joy": 0.94, "sadness": 0.02
            }}]},
        )
        result = emotion_detector("I am happy")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.90, "disgust": 0.02, "fear": 0.02,
                "joy": 0.03, "sadness": 0.03
            }}]},
        )
        result = emotion_detector("I am angry")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.02, "disgust": 0.02, "fear": 0.02,
                "joy": 0.04, "sadness": 0.90
            }}]},
        )
        result = emotion_detector("I am sad")
        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.02, "disgust": 0.02, "fear": 0.90,
                "joy": 0.03, "sadness": 0.03
            }}]},
        )
        result = emotion_detector("I am afraid")
        self.assertEqual(result["dominant_emotion"], "fear")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust(self, mock_post):
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {"emotionPredictions": [{"emotion": {
                "anger": 0.02, "disgust": 0.90, "fear": 0.02,
                "joy": 0.03, "sadness": 0.03
            }}]},
        )
        result = emotion_detector("That is disgusting")
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_bad_request(self, mock_post):
        mock_post.return_value = Mock(status_code=400)
        result = emotion_detector("bad request")
        self.assertIsNone(result["dominant_emotion"])

    def test_blank_input(self):
        result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
