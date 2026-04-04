from scipy.io import arff
import pandas as pd

# Load ARFF
data, meta = arff.loadarff('dataset_31_credit-g.arff')
df = pd.DataFrame(data)

# Convert bytes to strings
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Save as CSV
df.to_csv('credit-g.csv', index=False)

# Read CSV
df = pd.read_csv('credit-g.csv')

print(df.head())

#-------Missing values
print(df.isnull().sum())
df = df.dropna()

y = df['class']


# Selected features
numeric_features = ['duration', 'credit_amount', 'installment_commitment', 'age']
categorical_features = ['checking_status', 'credit_history', 'purpose']

X = df[numeric_features + categorical_features]

from sklearn.preprocessing import StandardScaler

# One-hot encode categorical features
X = pd.get_dummies(X, columns=categorical_features)

# Scale numeric features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

#---
from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)

#---Test Split (80/10/10)
from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Second split: 10% validation, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0
)

#---Train KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

best_k = 1
best_score = 0

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_val_pred = knn.predict(X_val)
    score = accuracy_score(y_val, y_val_pred)
    
    if score > best_score:
        best_score = score
        best_k = k

print("Best k:", best_k)

#--Confusion Matrix
from sklearn.metrics import confusion_matrix

# Train with best k
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

# Test predictions
y_test_pred = final_model.predict(X_test)

# Accuracy
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))