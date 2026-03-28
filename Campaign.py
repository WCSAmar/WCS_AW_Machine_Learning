import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('breast_cancer.csv')


df = df.dropna(axis=1, how='all').dropna()

# Drop ID column
if 'id' in df.columns:
    df = df.drop(columns=['id'])

#Feature selection
selected_features = [
    'diagnosis',
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean'
]

df = df[selected_features]

# Convert diagnosis: M = 1, B = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


scaler = StandardScaler()

# Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Scale features
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Combine back
df_processed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

df_processed.to_csv("data_refined.csv", index=False)


#Visualizations 
sns.pairplot(df_processed, hue='diagnosis')
#sns.pairplot(df_processed, height=1.5)
plt.show()

#Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_processed.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#Box Plots
plt.figure(figsize=(10, 6))
df_processed.drop(columns=['diagnosis']).boxplot()
plt.xticks(rotation=45)
plt.title("Box Plot of Features")
plt.show()

#Violin Plots
plt.figure(figsize=(10, 6))

for i, col in enumerate(X.columns):
    plt.subplot(2, 3, i+1)
    sns.violinplot(x=df['diagnosis'], y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()