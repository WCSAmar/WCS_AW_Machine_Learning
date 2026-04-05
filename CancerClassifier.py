import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("data_refined.csv")

#Diagnosed: M (Malignant), B (Benign)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])  # M=1, B=0

df = df.dropna()

#Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

#Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#
corr = X_scaled.corrwith(y)

# Choose features
threshold = 0.2
important_features = corr[abs(corr) > threshold].index.tolist()

print("Important features based on correlation:")
print(important_features)

X_reduced = X_scaled[important_features]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=0, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1111, random_state=0, stratify=y_train_val
)

# Same split for reduced dataset
Xr_train_val, Xr_test, yr_train_val, yr_test = train_test_split(
    X_reduced, y, test_size=0.1, random_state=0, stratify=y
)

Xr_train, Xr_val, yr_train, yr_val = train_test_split(
    Xr_train_val, yr_train_val, test_size=0.1111, random_state=0, stratify=yr_train_val
)

# KNN
param_grid = {'n_neighbors': list(range(1, 21))}
knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid, cv=5)
grid.fit(X_train, y_train)

best_k = grid.best_params_['n_neighbors']
print("Best k:", best_k)

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)

#Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

#SVC
svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train)

#Evaluation 
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"{name} Confusion Matrix:\n{cm}")

#Evaluate full feature set
print("\n=== FULL FEATURES ===")
evaluate(knn_model, X_test, y_test, "KNN")
evaluate(rf_model, X_test, y_test, "Random Forest")
evaluate(svc_model, X_test, y_test, "SVC")

#Train on reduced feature set
knn_model.fit(Xr_train, yr_train)
rf_model.fit(Xr_train, yr_train)
svc_model.fit(Xr_train, yr_train)

print("\n REDUCED FEATURES (Correlation)")
evaluate(knn_model, Xr_test, yr_test, "KNN")
evaluate(rf_model, Xr_test, yr_test, "Random Forest")
evaluate(svc_model, Xr_test, yr_test, "SVC")

#Alternative feature selection (SelectKBest)
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X_scaled, y)

selected_features = X.columns[selector.get_support()]
print("\nAlternative selected features (SelectKBest):")
print(selected_features)

# Split again
Xn_train_val, Xn_test, yn_train_val, yn_test = train_test_split(
    X_new, y, test_size=0.1, random_state=0, stratify=y
)

Xn_train, Xn_val, yn_train, yn_val = train_test_split(
    Xn_train_val, yn_train_val, test_size=0.1111, random_state=0, stratify=yn_train_val
)

# Train models again
knn_model.fit(Xn_train, yn_train)
rf_model.fit(Xn_train, yn_train)
svc_model.fit(Xn_train, yn_train)

print("\nREDUCED FEATURES (SelectKBest)")
evaluate(knn_model, Xn_test, yn_test, "KNN")
evaluate(rf_model, Xn_test, yn_test, "Random Forest")
evaluate(svc_model, Xn_test, yn_test, "SVC")