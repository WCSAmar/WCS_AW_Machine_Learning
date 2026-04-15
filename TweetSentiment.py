import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

data = pd.read_csv("text_emotion.csv")

#Select ONLY 5 emotions
selected_emotions = ['happiness', 'sadness', 'anger', 'love', 'surprise']
data = data[data['sentiment'].isin(selected_emotions)]

#Text and labels
texts = data['content'].astype(str)
labels = data['sentiment']

#Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

#Convert to categorical
labels_categorical = to_categorical(labels_encoded)

print("Classes:", label_encoder.classes_)
print("Number of classes:", len(label_encoder.classes_))

#Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

#Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)

#Maximum sequence length
max_len = max(len(seq) for seq in sequences)
print("Max Sequence Length:", max_len)

#Padding
X = pad_sequences(sequences, maxlen=max_len)

#Train/Test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels_categorical,
    test_size=0.3,
    random_state=42
)

#Build RNN model
model = Sequential()

#Embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=max_len))

#LSTM layers
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))

#Dense layer
model.add(Dense(100, activation='relu'))

#Dropout
model.add(Dropout(0.5))

#Output layer (5 classes)
model.add(Dense(5, activation='softmax'))

#Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Train model
history = model.fit(
    X_train,
    y_train,
    batch_size=256,
    epochs=10,
    validation_data=(X_test, y_test)
)

#Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)