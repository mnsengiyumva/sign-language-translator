"""
=============================================================================
recognize.py — Step 3 of the Sign Language Translator Pipeline
=============================================================================

WHAT THIS SCRIPT DOES:
    This is the live recognition script — the payoff of Steps 1 and 2.

    It opens your Mac webcam, runs MediaPipe on every frame to detect hand
    landmarks, feeds those 63 numbers into the trained model, and displays
    the predicted ASL letter on screen in real time.

    It also includes a simple word-building feature:
    - Hold a letter steady for 1 second → added to current word
    - Press SPACE → current word added to sentence
    - Press BACKSPACE → delete last letter
    - Press Q → quit

HOW TO RUN:
    python recognize.py

REQUIREMENTS:
    pip install mediapipe opencv-python numpy tensorflow
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import collections

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH     = "models/asl_model.keras"
LABEL_MAP_PATH = "models/label_map.npy"

# Model must be at least this confident to show a prediction (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.85

# Seconds you must hold a letter steady before it gets added to the word
HOLD_DURATION_SECONDS = 1.0

# Number of recent frames to consider when smoothing the prediction
# Larger = more stable but slightly slower to react
SMOOTHING_BUFFER_SIZE = 10

# =============================================================================
# LOAD MODEL AND LABEL MAP
# =============================================================================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# label_map is an array like ["A", "B", ..., "Z", "del", "nothing", "space"]
# The model outputs a number (e.g. 0) and we look up the letter here
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True)
print(f"Model loaded. Recognising {len(label_map)} classes.")

# =============================================================================
# MEDIAPIPE SETUP
# =============================================================================

mp_hands  = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# static_image_mode=False = video mode (faster, uses tracking between frames)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# =============================================================================
# WORD BUILDING STATE
# =============================================================================

# Stores the last N predictions for smoothing — prevents flickering output
prediction_buffer = collections.deque(maxlen=SMOOTHING_BUFFER_SIZE)

current_word     = ""   # letters collected in the current word
sentence         = []   # list of completed words
hold_start_time  = None # when we started holding the current letter
last_added_letter = ""  # avoids adding the same letter twice in a row

# =============================================================================
# HELPERS
# =============================================================================

def get_smoothed_prediction():
    """
    Returns the most common prediction in the recent buffer and a confidence
    score. This prevents the displayed letter flickering between similar signs.
    """
    if not prediction_buffer:
        return None, 0.0
    counts = collections.Counter(prediction_buffer)
    most_common_letter, count = counts.most_common(1)[0]
    smoothed_confidence = count / len(prediction_buffer)
    return most_common_letter, smoothed_confidence


def draw_text_with_background(frame, text, position, font_scale=1.0,
                               color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Draws text on a frame with a filled rectangle behind it so the text
    is always readable regardless of the background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_h - 5),
                  (x + text_w + 5, y + baseline + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_hold_progress(frame, progress, x, y, width=200):
    """
    Draws a horizontal progress bar showing how long the user has held
    the current letter. When full, the letter gets added to the word.
    """
    bar_height = 12
    cv2.rectangle(frame, (x, y), (x + width, y + bar_height), (80, 80, 80), -1)
    filled_width = int(width * min(progress, 1.0))
    cv2.rectangle(frame, (x, y), (x + filled_width, y + bar_height), (0, 220, 100), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + bar_height), (200, 200, 200), 1)


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    global current_word, hold_start_time, last_added_letter

    # Open the default webcam (try index 1 if 0 does not work on your Mac)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Check: System Settings → Privacy & Security → Camera")
        return

    print("\nWebcam open. Show your hand and sign!")
    print("SPACE = add word | BACKSPACE = delete letter | Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        # Flip horizontally so it acts like a mirror — more natural for signing
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape

        # ------------------------------------------------------------------
        # HAND DETECTION
        # ------------------------------------------------------------------

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result    = hands.process(frame_rgb)

        predicted_letter = None
        confidence       = 0.0

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            # Draw the hand skeleton on screen
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Extract the 63 landmark values (same format as extract_landmarks.py)
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            # Reshape to (1, 63) — model expects a batch of 1 sample
            input_data  = np.array(landmark_list, dtype=np.float32).reshape(1, -1)

            # Run the model — output is an array of probabilities, one per class
            predictions = model.predict(input_data, verbose=0)[0]
            confidence  = float(np.max(predictions))
            class_index = int(np.argmax(predictions))

            if confidence >= CONFIDENCE_THRESHOLD:
                predicted_letter = label_map[class_index]
                prediction_buffer.append(predicted_letter)
            else:
                prediction_buffer.append(None)
        else:
            # No hand in frame — clear the buffer and reset hold timer
            prediction_buffer.clear()
            hold_start_time = None

        # ------------------------------------------------------------------
        # SMOOTHING + WORD BUILDING
        # ------------------------------------------------------------------

        smoothed_letter, smoothed_confidence = get_smoothed_prediction()

        # Ignore the "nothing" class — means no meaningful sign is shown
        if smoothed_letter == "nothing":
            smoothed_letter = None

        hold_progress = 0.0

        if smoothed_letter and smoothed_letter not in ("nothing", "space", "del"):
            if hold_start_time is None or smoothed_letter != last_added_letter:
                hold_start_time = time.time()

            hold_elapsed  = time.time() - hold_start_time
            hold_progress = hold_elapsed / HOLD_DURATION_SECONDS

            # Letter held long enough — add it to the current word
            if hold_elapsed >= HOLD_DURATION_SECONDS:
                if smoothed_letter != last_added_letter:
                    current_word      += smoothed_letter
                    last_added_letter  = smoothed_letter
                    hold_start_time    = time.time()
                    print(f"Added: {smoothed_letter} | Word: {current_word}")
        else:
            hold_start_time = None

        # ------------------------------------------------------------------
        # KEYBOARD INPUT
        # ------------------------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            if current_word:
                sentence.append(current_word)
                print(f"Word: '{current_word}' | Sentence: {' '.join(sentence)}")
                current_word      = ""
                last_added_letter = ""
        elif key in (8, 127):   # backspace
            if current_word:
                current_word      = current_word[:-1]
                last_added_letter = ""

        # ------------------------------------------------------------------
        # DRAW UI
        # ------------------------------------------------------------------

        # Predicted letter + confidence
        if smoothed_letter and smoothed_confidence >= CONFIDENCE_THRESHOLD:
            draw_text_with_background(
                frame,
                f"Sign: {smoothed_letter}  ({smoothed_confidence*100:.0f}%)",
                (20, 60), font_scale=1.4,
                color=(0, 255, 120), bg_color=(20, 20, 20)
            )
            # Hold progress bar
            draw_hold_progress(frame, hold_progress, x=20, y=80)
        else:
            draw_text_with_background(
                frame, "No sign detected",
                (20, 60), font_scale=1.0,
                color=(100, 100, 100), bg_color=(20, 20, 20)
            )

        # Current word
        draw_text_with_background(
            frame, f"Word: {current_word}_",
            (20, frame_h - 80), font_scale=1.2,
            color=(255, 220, 50), bg_color=(30, 30, 30)
        )

        # Full sentence
        sentence_display = " ".join(sentence)
        if len(sentence_display) > 50:
            sentence_display = "..." + sentence_display[-47:]
        draw_text_with_background(
            frame, f"Sentence: {sentence_display}",
            (20, frame_h - 30), font_scale=0.8,
            color=(200, 200, 255), bg_color=(30, 30, 30)
        )

        # Controls
        draw_text_with_background(
            frame, "Q=Quit | SPACE=Word | BKSP=Delete",
            (frame_w - 420, 30), font_scale=0.55,
            color=(180, 180, 180), bg_color=(20, 20, 20)
        )

        cv2.imshow("ASL Sign Language Translator - Phase 1", frame)

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\nFinal sentence:", " ".join(sentence) if sentence else "(empty)")
    print("Session ended.")


if __name__ == "__main__":
    main()
