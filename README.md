# 🤟 Sign Language Translator

A real-time AI-powered sign language translation app that bridges communication between the Deaf and hearing communities — no interpreter needed.

---

## 🌟 What It Does

**Deaf → Hearing**
The app reads sign language gestures from a webcam, recognizes them using a trained AI model, and outputs both **text and audio** in real time.

**Hearing → Deaf**
A hearing person speaks or types, and the app generates a **human-like avatar** that performs the corresponding signs.

---

## 🚀 Features

- 🎥 Real-time hand gesture recognition via webcam
- 🤖 AI model trained on the ASL Alphabet dataset (87,000+ images)
- 🔊 Text-to-speech audio output for recognized signs
- 🧍 Animated avatar for sign generation (coming in Phase 4)
- 🌐 Multilingual support — English, French, Chinese, and more (coming in Phase 3)
- 💡 Runs on standard hardware — no expensive equipment needed

---

## 🏗️ Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | ASL letter recognition via webcam (landmark-based model) | 🔨 In Progress |
| 2 | React frontend with live camera UI | ⏳ Planned |
| 3 | Multilingual translation via LLM API | ⏳ Planned |
| 4 | 3D signing avatar for hearing → deaf direction | ⏳ Planned |
| 5 | Speech input via Whisper (voice → signs) | ⏳ Planned |

---

## 🧠 How It Works

```
Webcam Feed
    ↓
MediaPipe (extracts 21 hand landmark points per frame)
    ↓
Trained TensorFlow Classifier (predicts ASL letter)
    ↓
Text Output + Text-to-Speech Audio
```

Instead of training on raw images (which requires heavy GPU), the model trains on **hand landmark coordinates** extracted by MediaPipe. This makes it:
- Fast to train on a standard laptop
- Robust to different lighting conditions and backgrounds
- Accurate with less data

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Hand Detection | MediaPipe |
| AI Model | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Backend | Python 3.11 + FastAPI |
| Frontend (Phase 2) | React + TypeScript |
| 3D Avatar (Phase 4) | Three.js |
| Speech-to-Text | OpenAI Whisper |
| Text-to-Speech | pyttsx3 / ElevenLabs |
| Translation | Claude API / GPT-4o |

---

## 📁 Project Structure

```
sign-language-translator/
├── data/
│   ├── raw/                  # Raw dataset images
│   └── landmarks/            # Extracted landmark CSV files
├── models/                   # Saved trained model files
├── notebooks/                # Jupyter notebooks for experiments
├── extract_landmarks.py      # Step 1: Extract landmarks from dataset
├── train_model.py            # Step 2: Train the ASL classifier
├── recognize.py              # Step 3: Live webcam recognition
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites
- Mac (Apple Silicon or Intel)
- Python 3.11+
- Webcam

### Installation

```bash
# Clone the repo
git clone https://github.com/mnsengiyumva/sign-language-translator.git
cd sign-language-translator

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
# Step 1 — Extract landmarks from the dataset
python extract_landmarks.py

# Step 2 — Train the model
python train_model.py

# Step 3 — Start live recognition
python recognize.py
```

---

## 📦 Dataset

This project uses the **ASL Alphabet Dataset** from Kaggle:
- 87,000 images across 29 classes (A–Z + space, delete, nothing)
- [View on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

Download it and place the contents inside `data/raw/` before running `extract_landmarks.py`.

---

## 🗺️ Roadmap

- [x] Project structure and environment setup
- [ ] Landmark extraction pipeline
- [ ] ASL letter classification model (Phase 1)
- [ ] Text-to-speech output (Phase 1.5)
- [ ] React frontend with webcam UI (Phase 2)
- [ ] Multilingual translation (Phase 3)
- [ ] Signing avatar (Phase 4)
- [ ] Voice input via Whisper (Phase 5)

---

## 🤝 Motivation

Millions of Deaf and hard-of-hearing people face daily communication barriers with hearing individuals. Professional sign language interpreters are scarce and expensive. This project aims to make real-time, accessible sign language translation available to anyone with a camera and a device — for free.

---

## 👤 Author

**Mico Nsengiyumva**
Computer Science Student @ University of Prince Edward Island (UPEI) · Class of 2027
[GitHub](https://github.com/mnsengiyumva)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
