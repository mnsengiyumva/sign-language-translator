"""
=============================================================================
train_model.py — Step 2 of the Sign Language Translator Pipeline
=============================================================================

WHAT THIS SCRIPT DOES:
    This script reads the landmark CSV file produced by extract_landmarks.py
    and trains a neural network to classify ASL hand signs.

    The CSV contains thousands of rows where each row is:
    "these 63 numbers → this letter"

    We show those examples to a neural network repeatedly (called epochs),
    and it gradually learns which number patterns correspond to which letters.

    Once trained, the model is saved to disk so recognize.py can load it
    and use it in real time without retraining.

EXPECTED INPUT:
    data/landmarks/landmarks.csv  (produced by extract_landmarks.py)

OUTPUT:
    models/asl_model.keras    — the trained model file
    models/label_map.npy      — maps class numbers back to letter names

HOW TO RUN:
    python train_model.py

REQUIREMENTS:
    pip install tensorflow scikit-learn numpy pandas
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# =============================================================================
# CONFIGURATION
# =============================================================================

LANDMARK_CSV    = "data/landmarks/landmarks.csv"
MODEL_OUTPUT    = "models/asl_model.keras"
LABEL_MAP_OUTPUT = "models/label_map.npy"

# How many full passes through the training data
# EarlyStopping below will stop before this if the model stops improving
EPOCHS = 50

# How many rows the model processes at once before updating its weights
BATCH_SIZE = 32

# 20% of data held back for honest testing — never seen during training
TEST_SPLIT = 0.2

# =============================================================================
# STEP 1: LOAD AND PREPARE THE DATA
# =============================================================================

def load_data():
    """
    Reads the landmark CSV, separates features (the 63 numbers) from labels
    (the letter), and encodes the letters as numbers for the neural network.

    Neural networks work with numbers, not strings — so "A" becomes 0,
    "B" becomes 1, etc. We save the mapping so we can reverse it later
    in recognize.py.
    """

    print("\n[1/4] Loading landmark data...")

    df = pd.read_csv(LANDMARK_CSV)
    print(f"      Loaded {len(df)} rows across {df['label'].nunique()} classes")
    print(f"      Classes: {sorted(df['label'].unique())}")

    # X = the 63 landmark numbers (features)
    # y = the letter label
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    # Encode string labels ("A", "B" ...) as integers (0, 1, 2 ...)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)

    # Save the label map so recognize.py can convert prediction numbers back to letters
    os.makedirs(os.path.dirname(LABEL_MAP_OUTPUT), exist_ok=True)
    np.save(LABEL_MAP_OUTPUT, encoder.classes_)
    print(f"      Label map saved to: {LABEL_MAP_OUTPUT}")

    # Convert integer labels to one-hot encoding
    # e.g. label 2 of 29 classes → [0, 0, 1, 0, 0, ..., 0]
    num_classes = len(encoder.classes_)
    y_onehot = to_categorical(y_encoded, num_classes=num_classes)

    return X, y_onehot, num_classes


# =============================================================================
# STEP 2: SPLIT INTO TRAINING AND TEST SETS
# =============================================================================

def split_data(X, y):
    """
    Splits the data into a training set (what the model learns from) and
    a test set (used only at the end to give an honest accuracy score).

    The test set is kept completely hidden from the model during training.
    """

    print("\n[2/4] Splitting into train/test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=42,   # fixed seed so results are reproducible
        stratify=y         # ensures each letter is equally represented in both sets
    )

    print(f"      Training samples : {len(X_train)}")
    print(f"      Test samples     : {len(X_test)}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 3: BUILD THE NEURAL NETWORK
# =============================================================================

def build_model(input_size, num_classes):
    """
    Defines the neural network architecture.

    This is a feedforward neural network (Multi-Layer Perceptron):
    - Input  : 63 numbers (the landmark coordinates)
    - Hidden : layers that learn increasingly abstract patterns
    - Output : probabilities for each ASL class — highest one wins

    Layer explanations:
        Dense(256)         — 256 neurons that learn patterns from input
        BatchNormalization — keeps training stable by normalising outputs
        Dropout(0.3)       — randomly disables 30% of neurons each step
                             to prevent overfitting (memorising the data)
        softmax            — converts raw scores into probabilities (sum = 1)
    """

    print("\n[3/4] Building neural network...")

    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_size,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        # Output layer — one neuron per ASL class
        Dense(num_classes, activation="softmax")
    ])

    # adam      = smart gradient descent algorithm (standard choice)
    # loss      = standard loss function for multi-class classification
    # accuracy  = the metric we care about most
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


# =============================================================================
# STEP 4: TRAIN THE MODEL
# =============================================================================

def train_model(model, X_train, X_test, y_train, y_test):
    """
    Trains the neural network and saves the best version automatically.

    Two callbacks run during training:
    - EarlyStopping   : stops if validation accuracy stops improving,
                        preventing wasted time and overfitting
    - ModelCheckpoint : saves the model to disk every time it improves
    """

    print("\n[4/4] Training the model...")

    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,               # stop after 10 epochs with no improvement
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_OUTPUT,
            monitor="val_accuracy",
            save_best_only=True,       # only save when accuracy improves
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    return history


# =============================================================================
# EVALUATE
# =============================================================================

def evaluate(model, X_test, y_test):
    """
    Runs the trained model on the test set and prints final accuracy.
    This is the honest score — data the model has never seen before.
    """

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Test Accuracy  : {accuracy * 100:.2f}%")
    print(f"  Test Loss      : {loss:.4f}")
    print(f"  Model saved to : {MODEL_OUTPUT}")
    print(f"  Labels saved to: {LABEL_MAP_OUTPUT}")
    print("=" * 60)

    if accuracy >= 0.90:
        print("  Model is ready for live recognition (>= 90% accuracy)")
    elif accuracy >= 0.75:
        print("  Decent accuracy — consider increasing IMAGES_PER_CLASS")
    else:
        print("  Low accuracy — check your dataset or increase IMAGES_PER_CLASS")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("=" * 60)
    print("  ASL Model Training - Starting")
    print("=" * 60)

    X, y, num_classes = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model(input_size=X.shape[1], num_classes=num_classes)
    train_model(model, X_train, X_test, y_train, y_test)
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
