import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


#Breast Cancer Classification
print("\nBREAST CANCER CLASSIFICATION")
print("============================")

#Load dataset
data = pd.read_csv("data_refined.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#Split (80/10/10)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#Sklearn MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print("\n--- Sklearn MLP ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Keras Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)

loss, acc = model.evaluate(X_test, y_test, verbose=0)

y_pred_keras = (model.predict(X_test) > 0.5).astype(int)

print("\n--- Keras Model ---")
print("Accuracy:", acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_keras))

#Insurance Regression
print("\nINSURANCE REGRESSION")
print("====================")

#Load dataset
data = pd.read_csv("insurance.csv")

#Convert categorical → numeric
data = pd.get_dummies(data, drop_first=True)

X = data.drop("charges", axis=1)
y = data["charges"]

#Split (80/10/10)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#Sklearn MLP Regressor
mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp_reg.fit(X_train, y_train)

y_pred = mlp_reg.predict(X_test)

print("\nSklearn MLP Regressor")
print("R2 Score:", r2_score(y_test, y_pred))

#Keras Model
model = Sequential()
#model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)

y_pred_keras = model.predict(X_test)

print("\nKeras Regression")
print("R2 Score:", r2_score(y_test, y_pred_keras))