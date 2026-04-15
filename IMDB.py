import os
import warnings

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Optional: disable oneDNN message
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Suppress Python warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing import sequence
from keras.datasets import imdb

#Load dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    path="imdb.npz",
    maxlen=130,
    num_words=6000
)

x_train = sequence.pad_sequences(x_train, maxlen=130)
x_test = sequence.pad_sequences(x_test, maxlen=130)

model = Sequential()

#model.add(Embedding(input_dim=6000,
#                    output_dim=128,
#                    input_length=130))
model.add(Embedding(input_dim=6000,
                    output_dim=128
                    ))
model.add(LSTM(32))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(1, activation='sigmoid'))

#Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#
model.fit(x_train, y_train,
          epochs=5,
          batch_size=100,
          validation_data=(x_test, y_test),
          verbose=1)

model