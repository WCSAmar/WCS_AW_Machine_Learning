import pandas as pd
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv('results.csv')

#Cleaning missing entries
print("Missing values before cleaning:\n", df.isnull().sum())
df_cleaned = df.dropna()

print("\nMissing values after cleaning:\n", df.isnull().sum())

#Number of tuples
print("\nNumber of tuples:", len(df_cleaned))

#Number of unique tournaments
print("Number of tournaments:", df_cleaned['tournament'].unique())

#Convert date column to datetime
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])

#Matches played in 2018
matches_2018 = df_cleaned[df_cleaned['date'].dt.year == 2018]
print("Matches in 2018:", len(matches_2018))

#Home team results
home_wins = (df_cleaned['home_score'] > df_cleaned['away_score']).sum()
home_losses = (df_cleaned['home_score'] < df_cleaned['away_score']).sum()
draws = (df_cleaned['home_score'] == df_cleaned['away_score']).sum()

print("\nHome wins:", home_wins)
print("Home losses:", home_losses)
print("Draws:", draws)

#Pie chart for match outcomes
labels = ['Home Wins', 'Home Losses', 'Draws']
values = [home_wins, home_losses, draws]

plt.figure()
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title("Match Outcomes Distribution")
plt.show()

#Pie chart for neutral matches (one line)
df['neutral'].value_counts().plot.pie(autopct='%1.1f%%', title="Neutral Matches Distribution")
plt.ylabel('')
plt.show()

#Unique teams (home + away combined)
unique_teams = pd.unique(df[['home_team', 'away_team']].values.ravel())

print("\nNumber of unique teams:", len(unique_teams))

 