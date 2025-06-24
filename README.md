# Emotion-Classification-on-Speech-Data
# 🎧 Emotion Classification from Speech using CNN + BiLSTM + Transformer

This project performs **emotion classification** on audio files using a hybrid deep learning model that combines **1D Convolutional Neural Networks (CNNs)** with **BiLSTM + Transformer blocks**. The model is trained on the **RAVDESS** dataset and supports real-time predictions via a test script and (optionally) a web app.

---

## 📁 Dataset

We use the [RAVDESS dataset](https://zenodo.org/record/1188976) which contains **emotional speech and song recordings** from 24 actors. Emotions included:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

Only `.wav` files from `Audio_Speech_Actors_01-24` and `Audio_Song_Actors_01-24` are used.

---

## ⚙️ Preprocessing & Feature Extraction

Each audio file is processed as follows:

- Trim silence
- Extract:
  - **Zero Crossing Rate (ZCR)**
  - **Root Mean Square Energy (RMSE)**
  - **MFCCs** (Mel-Frequency Cepstral Coefficients, 30)
- Augmentations:
  - Noise injection
  - Pitch shifting
  - Pitch + noise combined

All extracted features are saved to CSV for reuse.

---

## 🧠 Model Architecture

The model is a **dual-branch hybrid network**:

### 🔹 Branch 1: CNN
- 5 layers of 1D Convolutions
- Batch Normalization, MaxPooling, Dropout
- Captures **local temporal features** from MFCC sequences

### 🔹 Branch 2: BiLSTM + Transformer
- Bidirectional LSTM (64 units)
- 3 stacked custom Transformer blocks
- Captures **sequential and long-range dependencies**

### 🔹 Output
- Concatenation of both branches
- Dense + Dropout + Layer Normalization
- Final output layer: `Softmax` over 8 emotion classes

---

## 📊 Evaluation

| Metric        | Value        |
|---------------|--------------|
| Train Accuracy| ✅ 98.4%      |
| Val Accuracy  | ✅ 92.6%      |
| Loss Function | `categorical_crossentropy` |
| Optimizer     | `adam`       |

Callbacks:
- `ModelCheckpoint`
- `EarlyStopping`
- `ReduceLROnPlateau`

---

## 🧪 How to Use

### ✅ Notebook Training
Use `emotion_classification.ipynb` to:
- Preprocess dataset
- Train the model
- Save weights to `.keras`

### ✅ Run Inference via Script

```bash
python test_model.py path/to/audio.wav
