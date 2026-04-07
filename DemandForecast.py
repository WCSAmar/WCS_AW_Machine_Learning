# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("Historical Product Demand.csv")

df = df.dropna()

X = df.drop(columns=['Date', 'Order_Demand'])
y = df['Order_Demand']

X = pd.get_dummies(X, drop_first=True)

X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns='Order_Demand')
y = data['Order_Demand']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42
)  

best_k = 0
best_r2 = -np.inf

for k in range(1, 21):  # Try k = 1 to 20
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    if r2 > best_r2:
        best_r2 = r2
        best_k = k

print(f"Best k: {best_k} with R² on validation set: {best_r2:.4f}")

X_final_train = np.vstack((X_train, X_val))
y_final_train = pd.concat([y_train, y_val])

final_knn = KNeighborsRegressor(n_neighbors=best_k)
final_knn.fit(X_final_train, y_final_train)

y_test_pred = final_knn.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)
print(f"R² on test set: {r2_test:.4f}")
