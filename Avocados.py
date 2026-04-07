import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

 
df = pd.read_csv("avocado.csv") 
print(df)

print("Missing values before drop:\n", df.isnull().sum())
df = df.dropna()
print("DROP MISSING VALUES:\n",df)

y = df["AveragePrice"]


X = df.drop(columns=["region", "Date", "AveragePrice"], errors='ignore')


X = pd.get_dummies(X, drop_first=True)
 
X = X.apply(pd.to_numeric, errors='coerce')
print("USING DUMMIES NOW NUMERICAL:\n",X)

data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=["AveragePrice"])
y = data["AveragePrice"]


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=0
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


best_k = None
best_score = -float('inf')

for k in range(1, 21):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict(X_val)
    score = r2_score(y_val, y_val_pred)
    
    if score > best_score:
        best_score = score
        best_k = k

print("Best k:", best_k)


final_model = KNeighborsRegressor(n_neighbors=best_k)
final_model.fit(X_train, y_train)


y_test_pred = final_model.predict(X_test)
final_r2 = r2_score(y_test, y_test_pred)

print("Final R-squared score on test set:", final_r2)
 