import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

#Load data
df = pd.read_csv('flavors_of_cacao.csv')

# Columns renamed
df.columns = ['Company', 'SpecificBeanOrigin',
              'REF', 'ReviewDate', 'CocoaPercent', 'CompanyLocation',
              'Rating', 'BeanType', 'BroadBeanOrigin']

 
#Cleaning missing entries
df_cleaned = df.dropna()

print("Number of tuples:",len(df_cleaned))
print("Unique companies:", df_cleaned['Company'].unique())

# Reviews in 2013
print("Reviews in 2013:", len(df_cleaned[df_cleaned['ReviewDate'] == 2013]))
#Missing values in BeanType column
print("Missing BeanType:", df['BeanType'].str.isspace().sum())

#Rating column with a histogram
plt.hist(df_cleaned['Rating'], bins=4)
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

#Observation: Ratings are mostly between 3.0 and 4.0
df_cleaned['CocoaPercent'] = df_cleaned['CocoaPercent'].str.rstrip('%').astype(float).astype(int)
print(df_cleaned)

#Scatter Plot
plt.scatter(df_cleaned['CocoaPercent'], df_cleaned['Rating'], alpha=0.1)
plt.xlabel("Cocoa Percent")
plt.ylabel("Rating")
plt.title("Cocoa % vs Rating")
plt.show()

#Normalization
scaler = StandardScaler()
df_cleaned['Normalized_Rating'] = scaler.fit_transform(df_cleaned[['Rating']])
print(df_cleaned[['Rating', 'Normalized_Rating']].head())
 
#Companies ordered by their average score
company_avg = df_cleaned.groupby('Company')['Rating'].mean().sort_values(ascending=False)
print(company_avg.head())

enc = OrdinalEncoder()
df_cleaned
df_cleaned[['Company_encoded', 'Location_encoded']] = enc.fit_transform(
    df_cleaned[['Company', 'CompanyLocation']]
)
print("\nEncoded columns:\n", df_cleaned[['Company', 'Company_encoded', 'CompanyLocation', 'Location_encoded']].head())