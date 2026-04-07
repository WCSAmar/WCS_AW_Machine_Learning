import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# 1. Load dataset
df = pd.read_csv("avocado.csv")  # make sure file is in your working directory
print(df)
# 2. Check and drop missing values
print("Missing values before drop:\n", df.isnull().sum())
df = df.dropna()
print("DROP MISSING VALUES:\n",df)
# 3. Define target variable
y = df["AveragePrice"]

# 4. Drop unnecessary columns (region, date, target)
X = df.drop(columns=["region", "Date", "AveragePrice"], errors='ignore')

# 5. Encode categorical variables
X = pd.get_dummies(X, drop_first=True)
print("USING DUMMIES:\n",X)
# 6. Ensure all data is numeric (important!)
X = X.apply(pd.to_numeric, errors='coerce')
print("USING DUMMIES NOW NUMERICAL:\n",X)
# 7. Remove any rows with NaNs after encoding
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=["AveragePrice"])
y = data["AveragePrice"]

# 8. Split data: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=0
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0
)

# 9. Scale features (VERY important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 10. Find best k using validation set
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

# 11. Train final model
final_model = KNeighborsRegressor(n_neighbors=best_k)
final_model.fit(X_train, y_train)

# 12. Evaluate on test set
y_test_pred = final_model.predict(X_test)
final_r2 = r2_score(y_test, y_test_pred)

print("Final R-squared score on test set:", final_r2)
 