
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("insurance.csv")

print("Missing values:\n", df.isnull().sum())
df = df.dropna()

sns.pairplot(df)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

df.plot(kind='box', subplots=True, layout=(3,3), figsize=(12,10))
plt.tight_layout()
plt.show()

y = df['charges']

X = df.drop(columns=['charges'])

X = pd.get_dummies(X, drop_first=True)

X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns='charges')
y = data['charges']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

normalizer = MinMaxScaler()
X_scaled = normalizer.fit_transform(X_scaled)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42
)  # 80/10/10

X_train_full = np.vstack((X_train, X_val))
y_train_full = pd.concat([y_train, y_val])

#Decision Tree
dt_params = {
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [3, 5, 10, None]
}

dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=0),
                       dt_params, cv=5)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_

#Random Forest
rf_params = {
    'n_estimators': [50, 100],
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [5, 10, None]
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=0),
                       rf_params, cv=5)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

#SVR
svr = SVR(kernel='rbf', C=100, gamma='scale')
svr.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n{name} Results:")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return r2, mse, mae

dt_best.fit(X_train_full, y_train_full)
rf_best.fit(X_train_full, y_train_full)
svr.fit(X_train_full, y_train_full)

dt_results = evaluate_model(dt_best, X_test, y_test, "Decision Tree")
rf_results = evaluate_model(rf_best, X_test, y_test, "Random Forest")
svr_results = evaluate_model(svr, X_test, y_test, "SVR")
