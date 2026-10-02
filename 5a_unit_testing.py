import unittest
from unittest.mock import patch, Mock

from EmotionDetection.emotion_detection import emotion_detector

class TestEmotionDetector(unittest.TestCase):
    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger(self, mock_post):
        self._set_response(mock_post)
        result = emotion_detector("I hate this.")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust(self, mock_post):
        self._set_response(mock_post, disgust=0.9, anger=0.1)
        result = emotion_detector("This is disgusting.")
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear(self, mock_post):
        self._set_response(mock_post, fear=0.9)
        result = emotion_detector("I am afraid.")
        self.assertEqual(result["dominant_emotion"], "fear")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy(self, mock_post):
        self._set_response(mock_post, joy=0.9)
        result = emotion_detector("I am very happy.")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness(self, mock_post):
        self._set_response(mock_post, sadness=0.9)
        result = emotion_detector("I feel sad.")
        self.assertEqual(result["dominant_emotion"], "sadness")

    @staticmethod
    def _set_response(mock_post, anger=0.2, disgust=0.1, fear=0.1, joy=0.1, sadness=0.1):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": anger, "disgust": disgust, "fear": fear,
                    "joy": joy, "sadness": sadness
                }
            }]
        }
        mock_response.text = str(mock_response.json.return_value)
        mock_post.return_value = mock_response

if __name__ == "__main__":
    unittest.main()
