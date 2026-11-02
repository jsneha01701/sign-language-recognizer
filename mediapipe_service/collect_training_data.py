import cv2
import numpy as np
from pathlib import Path
import time
from extractor import extract_landmarks_from_frame
from sign_config import GESTURES, SAMPLES_PER_GESTURE, SEQUENCE_LENGTH

DATA_DIR = Path("data/live_landmarks")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_PER_SAMPLE = SEQUENCE_LENGTH

print("ISL starter-vocabulary webcam collector")
print(f"Signs: {', '.join(GESTURES)}")
print(f"Target: {SAMPLES_PER_GESTURE} samples per sign, {FRAMES_PER_SAMPLE} frames per sample")
print("Perform the displayed sign throughout each recording.")
print("For REST, lower both hands and wait in a neutral position.")
print("SPACE = record | Q = next sign | ESC = quit")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    cap.release()
    raise RuntimeError("Could not open webcam. Check that it is connected and not in use.")

total_samples = 0
exit_collection = False
try:
    for gesture_index, gesture in enumerate(GESTURES, start=1):
        if exit_collection:
            break

        existing = list(DATA_DIR.glob(f"{gesture}_*.npy"))
        sample_count = len(existing)
        next_sample = max(
            (int(path.stem.rsplit("_", 1)[1]) for path in existing),
            default=0,
        ) + 1
        print(f"\n[{gesture_index}/{len(GESTURES)}] {gesture.upper()}: {sample_count}/{SAMPLES_PER_GESTURE}")
        skip_gesture = False

        while sample_count < SAMPLES_PER_GESTURE and not skip_gesture and not exit_collection:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam.")

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Sign: {gesture.upper()}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Sample: {sample_count}/{SAMPLES_PER_GESTURE}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE=record  Q=next  ESC=quit", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow("Collect ISL training samples", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                exit_collection = True
                break
            if key == ord("q"):
                skip_gesture = True
                break
            if key != ord(" "):
                continue

            sequence = []
            for frame_index in range(FRAMES_PER_SAMPLE):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam.")
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"RECORDING {frame_index + 1}/{FRAMES_PER_SAMPLE}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Collect ISL training samples", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    exit_collection = True
                    break
                sequence.append(extract_landmarks_from_frame(frame))

            if len(sequence) != FRAMES_PER_SAMPLE:
                print(f"Incomplete sample ({len(sequence)}/{FRAMES_PER_SAMPLE}); discarded.")
                continue

            array = np.asarray(sequence, dtype=np.float32)
            if not np.any(array):
                print("No landmarks detected; move into view and record this sign again.")
                continue

            output_path = DATA_DIR / f"{gesture}_{next_sample:03d}.npy"
            np.save(output_path, array)
            next_sample += 1
            sample_count += 1
            total_samples += 1
            print(f"Saved {output_path}")
            time.sleep(0.3)
finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"\nCollected {total_samples} samples in {DATA_DIR}.")
print("Next: python train.py, then open http://127.0.0.1:8000")
