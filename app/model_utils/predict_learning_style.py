# model_utils.py

import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load model and other components
model = load_model('app/model_utils/LearningStyleClassifier.h5')
with open('app/model_utils/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('app/model_utils/labelEncoder.pickle', 'rb') as handle:
    le = pickle.load(handle)


# Text cleaning function
def clean(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text


# Prediction function
def predict_learning_style(answers):
    cleaned_texts = [clean(answer) for answer in answers]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded_sequences = pad_sequences(sequences, maxlen=48, truncating='pre')

    # Predict learning style for each answer
    predictions = model.predict(padded_sequences)
    predicted_classes = le.inverse_transform(np.argmax(predictions, axis=1))
    confidences = np.max(predictions, axis=1)

    # Return predictions with confidence levels
    return [
        {"predicted_class": predicted_class, "confidence": float(confidence)}
        for predicted_class, confidence in zip(predicted_classes, confidences)
    ]
