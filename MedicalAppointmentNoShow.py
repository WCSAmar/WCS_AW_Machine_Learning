import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read dataset
dataset = pd.read_csv("KaggleV2-May-2016.csv")

#Drop missing values
dataset = dataset.dropna()

# Renaming columns to standard names
dataset = dataset.rename(columns={
    'Hipertension': 'Hypertension',
    'Handcap': 'Handicap'
})

features = ['Gender', 'Age', 'Scholarship', 'Hypertension', 'Diabetes', 
            'Alcoholism', 'Handicap', 'SMS_received']
X = dataset[features]
y = dataset['No-show']  # target column name

#Encode categorical features
categorical_features = ['Gender', 'Scholarship', 'Hypertension', 'Diabetes', 
                        'Alcoholism', 'Handicap', 'SMS_received']

le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le  # store encoders in case needed later

#Scale numeric features
scaler = StandardScaler()
X['Age'] = scaler.fit_transform(X[['Age']])

#Split dataset: 80% train, 10% val, 10% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1111, random_state=0, stratify=y_train_val
)
# Note: 0.1111 * 0.9 ≈ 0.1 overall validation set

#Decision Tree classifier
best_acc = 0
best_criterion = None
for criterion in ['gini', 'entropy']:
    dt_model = DecisionTreeClassifier(criterion=criterion, random_state=0)
    dt_model.fit(X_train, y_train)
    val_acc = dt_model.score(X_val, y_val)
    print(f"Decision Tree ({criterion}) validation accuracy: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        best_criterion = criterion

# Train final decision tree with best criterion on train+val
final_dt = DecisionTreeClassifier(criterion=best_criterion, random_state=0)
final_dt.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
y_pred_dt = final_dt.predict(X_test)

print("Decision Tree Test Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

#Random Forest classifier
n_estimators_list = [10, 50, 100, 200]
for n in n_estimators_list:
    rf_model = RandomForestClassifier(n_estimators=n, criterion='gini', random_state=0)
    rf_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    y_pred_rf = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_rf)
    cm = confusion_matrix(y_test, y_pred_rf)
    print(f"\nRandom Forest (n_estimators={n}) Test Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")