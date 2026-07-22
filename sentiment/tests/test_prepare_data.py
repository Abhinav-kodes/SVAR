import unittest
import os
import shutil
import tempfile
import pandas as pd
from sentiment.preprocessing.prepare_emoin_hindi import (
    map_emotion_taxonomy,
    parse_primary_emotion,
    build_context_windows,
    process_and_split_dataset
)


class TestDatasetPreprocessing(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_map_emotion_taxonomy(self):
        """Test fine-grained mapping to 6 core emotion classes."""
        self.assertEqual(map_emotion_taxonomy("annoyed"), "anger")
        self.assertEqual(map_emotion_taxonomy("guilty"), "sadness")
        self.assertEqual(map_emotion_taxonomy("apprehensive"), "fear")
        self.assertEqual(map_emotion_taxonomy("joy"), "happiness")
        self.assertEqual(map_emotion_taxonomy("disgusted"), "disgust")
        self.assertEqual(map_emotion_taxonomy("confident"), "neutral")

    def test_parse_primary_emotion(self):
        """Test primary emotion extraction based on highest intensity."""
        emotions = "joy, annoyed"
        intensities = "1, 3"
        primary = parse_primary_emotion(emotions, intensities)
        self.assertEqual(primary, "anger")

    def test_build_context_windows(self):
        """Test context window prepending for utterances in a dialogue."""
        data = {
            "dialogueId": [101, 101, 101],
            "utterance": ["Hello how can I help?", "My account is blocked.", "I am very annoyed."]
        }
        df = pd.DataFrame(data)
        processed = build_context_windows(df, window_size=2)

        self.assertIn("input_text", processed.columns)
        input_texts = processed["input_text"].tolist()
        self.assertEqual(input_texts[0], "Hello how can I help?")
        self.assertEqual(input_texts[1], "Hello how can I help? [SEP] My account is blocked.")
        self.assertEqual(
            input_texts[2],
            "Hello how can I help? [SEP] My account is blocked. [SEP] I am very annoyed."
        )

    def test_process_and_split_dataset(self):
        """Test processing and dialogue-level 80/10/10 splitting."""
        dialogues = []
        for d_id in range(10):
            for u_idx in range(4):
                dialogues.append({
                    "dialogueId": d_id,
                    "utterance": f"Utterance {u_idx} in dialogue {d_id}",
                    "emotions": "joy" if u_idx % 2 == 0 else "neutral",
                    "emoIntensity": "2"
                })
        df = pd.DataFrame(dialogues)

        splits = process_and_split_dataset(df, self.temp_dir, train_ratio=0.8, val_ratio=0.1)

        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)

        train_ids = set(splits["train"]["dialogueId"])
        val_ids = set(splits["val"]["dialogueId"])
        test_ids = set(splits["test"]["dialogueId"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "train.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "val.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test.csv")))


if __name__ == "__main__":
    unittest.main()
