import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, BatchNormalization,
    Conv1D, MaxPooling1D, Input, LayerNormalization,
    MultiHeadAttention, Lambda, LSTM, Bidirectional, concatenate
)

# 🎯 Transformer Block Definition
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

# 🎯 Build Model
def build_model(input_shape=(3456, 1)):
    input_cnn = Input(shape=input_shape)
    x = Conv1D(512, 5, padding='same', activation='relu')(input_cnn)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=5, strides=2, padding='same')(x)
    x = Conv1D(512, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=5, strides=2, padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(256, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=5, strides=2, padding='same')(x)
    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=5, strides=2, padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    x = Dropout(0.2)(x)
    x_cnn = Flatten()(x)

    input_bilstm = Input(shape=input_shape)
    y = Bidirectional(LSTM(64, return_sequences=False))(input_bilstm)
    y = Lambda(lambda x: tf.expand_dims(x, axis=1), output_shape=(1, 128))(y)
    for _ in range(3):
        y = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512)(y)
    y = Flatten()(y)

    combined = concatenate([x_cnn, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = LayerNormalization()(z)
    output = Dense(8, activation='softmax')(z)
    model = Model(inputs=[input_cnn, input_bilstm], outputs=output)
    return model

# 🎯 Load model and weights
model = build_model(input_shape=(3456, 1))
model.load_weights("Emotion_Model.weights.h5")

# 🎯 Emotion labels
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# 🎯 Audio feature preprocessing (raw waveform, padded to 3456 samples)
def extract_features(data, sr, target_length=3456):
    if len(data) > target_length:
        data = data[:target_length]
    else:
        data = np.pad(data, (0, target_length - len(data)))
    return data.reshape(1, -1, 1)  # shape (1, 3456, 1)

# 🎯 Preprocessing
def preprocess_audio(file_path):
    data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    features = extract_features(data, sr)
    return features, data, sr

# 🎯 Streamlit Interface
st.set_page_config(page_title="🎤 Emotion Detection from Speech")
st.title("🎤 Emotion Detection from Speech")
st.write("Upload a `.wav` file to detect the emotion.")

file = st.file_uploader("Choose a WAV audio file", type=["wav"])

if file is not None:
    with open("temp.wav", "wb") as f:
        f.write(file.read())
    st.audio("temp.wav")

    features, data, sr = preprocess_audio("temp.wav")

    # 🎯 Plot waveform
    st.subheader("📈 Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(data, sr=sr, ax=ax,color="blue")
    st.pyplot(fig)

    # 🎯 Plot mel spectrogram
    st.subheader("🎛️ Mel Spectrogram")
    fig, ax = plt.subplots()
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)

    # 🎯 Predict
    with st.spinner("Predicting..."):
        prediction = model.predict([features, features])[0]
        top_emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.subheader("🎯 Predicted Emotion:")
    st.success(f"{top_emotion.upper()} ({confidence:.2f}%)")
