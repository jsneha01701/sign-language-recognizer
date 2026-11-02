import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

try:
    from .model import LSTMClassifier
    from .sign_config import DISPLAY_WORDS, GESTURES, SEQUENCE_LENGTH
except ImportError:
    from model import LSTMClassifier
    from sign_config import DISPLAY_WORDS, GESTURES, SEQUENCE_LENGTH

CONFIDENCE_THRESHOLD = 0.60
VOTES_TO_CONFIRM = 3
REQUIRED_SIGNS = set(GESTURES)


class SentenceRecognizer:
    def __init__(self, model_dir: Path):
        model_path = model_dir / "model.pt"
        labels_path = model_dir / "labels_map.json"
        if not model_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError("Train the live-sign model with python train.py before recognition.")

        with labels_path.open(encoding="utf-8") as file:
            self.idx_to_label = json.load(file)
        if set(self.idx_to_label.values()) != REQUIRED_SIGNS:
            raise ValueError(
                "The trained model does not contain the complete ISL starter vocabulary. "
                "Collect samples for hello, i, need, help, water, thankyou, yes, no, and rest, "
                "then run python train.py."
            )

        self.model = LSTMClassifier(
            input_dim=258,
            hidden_dim=128,
            num_layers=2,
            num_classes=len(self.idx_to_label),
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        self.frames = deque(maxlen=SEQUENCE_LENGTH)
        self.votes = deque(maxlen=VOTES_TO_CONFIRM)
        self.words = []
        self.armed = True

    def reset(self):
        self.frames.clear()
        self.votes.clear()
        self.words.clear()
        self.armed = True

    def observe(self, landmarks):
        if len(landmarks) != 258:
            raise ValueError(f"Expected 258 landmark values, received {len(landmarks)}.")

        self.frames.append(landmarks)
        event = None
        confidence = None
        if len(self.frames) == SEQUENCE_LENGTH:
            sequence = np.asarray(self.frames, dtype=np.float32)
            with torch.no_grad():
                probabilities = torch.softmax(
                    self.model(torch.from_numpy(sequence[None, ...])), dim=-1
                )[0]
            class_index = int(torch.argmax(probabilities))
            confidence = float(probabilities[class_index])
            label = self.idx_to_label[str(class_index)]

            if confidence < CONFIDENCE_THRESHOLD:
                self.votes.clear()
            else:
                self.votes.append(label)
                if len(self.votes) == VOTES_TO_CONFIRM and len(set(self.votes)) == 1:
                    if label == "rest":
                        self.armed = True
                        self.votes.clear()
                    elif self.armed:
                        self.words.append(DISPLAY_WORDS[label])
                        event = label
                        self.armed = False
                        self.votes.clear()

        sentence = " ".join(self.words)
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        return {
            "sentence": sentence,
            "event": event,
            "confidence": confidence,
            "buffered_frames": len(self.frames),
            "sequence_length": SEQUENCE_LENGTH,
        }
