# Financial-News-Sentiment-Analysis-with-Deep-Learning
Performed Sentiment Analysis using Deep Learning with TensorFlow in Google Colab (T4 GPU) or a local environment. Extract training data from Hugging Face’s Twitter Financial News Sentiment dataset. Labels: 0 – Bearish, 1 – Bullish, 2 – Neutral. Trained the model and used PKL file for validation.

**The model is trained to classify tweets into three categories:**

**0 → Bearish
1 → Bullish
2 → Neutral**_

The dataset is sourced from Hugging Face (Hugging Face - zeroshot/twitter-financial-news-sentiment), and the training is performed in Google Colab (T4 GPU).

**Step 1: Environment Setup**

Open Google Colab →  Enable GPU support : Runtime → Change runtime type → Select GPU (T4 recommended)

**Step 2: Load the Dataset**

Dataset Source : Hugging Face - Twitter Financial News Sentiment

Libraries Installation

!pip install tensorflow datasets transformers pandas scikit-learn

Code to Load Data

import pandas as pd

splits = {'train': 'sent_train.csv', 'validation': 'sent_valid.csv'}

df_train = pd.read_csv("hf://datasets/zeroshot/twitter-financial-news-sentiment/" + splits["train"])
df_valid = pd.read_csv("hf://datasets/zeroshot/twitter-financial-news-sentiment/" + splits["validation"])

df_train.head()

**Step 3: Data Preprocessing**

Convert text labels to numerical format : Bearish → 0, Bullish → 1 & Neutral → 2

Tokenize and pad text sequences

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

label_map = {'Bearish': 0, 'Bullish': 1, 'Neutral': 2}

df_train.columns = ['text', 'label']
df_valid.columns = ['text', 'label']

df_train['label'] = df_train['label'].map(label_map)
df_valid['label'] = df_valid['label'].map(label_map)

tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df_train['text'])

X_train = tokenizer.texts_to_sequences(df_train['text'])
X_valid = tokenizer.texts_to_sequences(df_valid['text'])

X_train = pad_sequences(X_train, maxlen=100, padding='post', truncating='post')
X_valid = pad_sequences(X_valid, maxlen=100, padding='post', truncating='post')

y_train = np.array(df_train['label'])
y_valid = np.array(df_valid['label'])

**Step 4: Build Deep Learning Model**

Model Consists :

●	Embedding Layer for text representation

●	LSTM Layers for sequential data processing

●	Dense Layers for classification

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

**Step 5: Train the Model**

Training the model using 10 epochs with a batch size of 32
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=10,
    batch_size=32
)

**Step 6: Save and Export the Model**

Once training is completed, save the model in PKL format for validation

import pickle

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save('3681_sentiment_model_Kshitiz_Sharma.h5')

with open('3681_sentiment_model_Kshitiz_Sharma.pkl', 'wb') as f:
    pickle.dump(model, f)

Download the Model

Downloading the 3681_sentiment_model_Kshitiz_Sharma.pkl file for validation

from google.colab import files
files.download('3681_sentiment_model_Kshitiz_Sharma.pkl')


**Step 7: Load and Test the Model**

After saving, the model can be reloaded for inference

from tensorflow.keras.models import load_model

loaded_model = load_model('3681_sentiment_model_Kshitiz_Sharma.h5')

with open('tokenizer.pkl', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

def predict_sentiment(text):
    sequence = loaded_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = loaded_model.predict(padded_sequence)
    label_map_reverse = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    return label_map_reverse[np.argmax(prediction)]

test_text = input("Enter a Tweet: ")
print(predict_sentiment(test_text))

**Step 8: Model Validation**

●	The model predicts sentiment for a given tweet.

●	The trained model can be used for further testing.

