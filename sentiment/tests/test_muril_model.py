import unittest
import torch
import pandas as pd
from sentiment.models.dataset import EmotionDataset, compute_class_weights, EMOTION_LABEL2ID
from sentiment.models.muril_model import MultiTaskMuRIL


class TestMuRILDatasetAndModel(unittest.TestCase):

    def test_emotion_dataset_indexing(self):
        """Test EmotionDataset length, indexing, and tensor keys."""
        data = {
            "input_text": ["मुझे गुस्सा आ रहा है", "मैं बहुत उदास हूँ"],
            "mapped_label": ["anger", "sadness"],
            "emoIntensity": ["3", "2"]
        }
        df = pd.DataFrame(data)

        dataset = EmotionDataset(df, tokenizer=None, max_len=64)
        self.assertEqual(len(dataset), 2)

        sample = dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("attention_mask", sample)
        self.assertIn("emotion_label", sample)
        self.assertIn("sentiment_label", sample)
        self.assertIn("intensity", sample)

        self.assertEqual(sample["emotion_label"].item(), EMOTION_LABEL2ID["anger"])

    def test_compute_class_weights(self):
        """Test class weight calculation produces 6-dim tensor."""
        data = {
            "mapped_label": ["anger", "anger", "sadness", "neutral"]
        }
        df = pd.DataFrame(data)

        weights = compute_class_weights(df)
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (6,))

    def test_muril_model_forward_pass(self):
        """Test MultiTaskMuRIL forward pass tensor output dimensions."""
        model = MultiTaskMuRIL(model_name="google/muril-base-cased")

        batch_size = 2
        seq_len = 32
        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        outputs = model(input_ids, attention_mask=attention_mask)

        self.assertIn("emotion_logits", outputs)
        self.assertIn("sentiment_logits", outputs)
        self.assertIn("intensity_pred", outputs)

        self.assertEqual(outputs["emotion_logits"].shape, (batch_size, 6))
        self.assertEqual(outputs["sentiment_logits"].shape, (batch_size, 3))
        self.assertEqual(outputs["intensity_pred"].shape, (batch_size, 1))


if __name__ == "__main__":
    unittest.main()
