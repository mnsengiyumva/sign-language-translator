import os
import csv
import cv2
import mediapipe as mp
import numpy as np

DATASET_PATH = "data/raw/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = "data/landmarks/landmarks.csv"
IMAGES_PER_CLASS = 500

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None
    hand_landmarks = result.multi_hand_landmarks[0]
    landmark_list = []
    for landmark in hand_landmarks.landmark:
        landmark_list.extend([landmark.x, landmark.y, landmark.z])
    return landmark_list

def main():
    print("=" * 60)
    print("  ASL Landmark Extraction - Starting")
    print("=" * 60)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = [f"landmark_{i}" for i in range(63)] + ["label"]
        writer.writerow(header)

        class_folders = sorted(os.listdir(DATASET_PATH))
        total_written = 0
        total_skipped = 0

        for label in class_folders:
            folder_path = os.path.join(DATASET_PATH, label)
            if not os.path.isdir(folder_path):
                continue

            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if IMAGES_PER_CLASS is not None:
                image_files = image_files[:IMAGES_PER_CLASS]

            written = 0
            skipped = 0

            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                landmarks = extract_landmarks_from_image(image_path)
                if landmarks is None:
                    skipped += 1
                    continue
                writer.writerow(landmarks + [label])
                written += 1

            print(f"  [{label}] {written} saved, {skipped} skipped")
            total_written += written
            total_skipped += skipped

    hands.close()
    print("=" * 60)
    print(f"  Done! {total_written} rows saved to: {OUTPUT_CSV}")
    print(f"  {total_skipped} images skipped (no hand detected)")
    print("=" * 60)

if __name__ == "__main__":
    main()
