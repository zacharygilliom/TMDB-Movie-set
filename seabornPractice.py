import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ast
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set(style="ticks")

df = pd.read_csv("train.csv")
print(df.head())

print(df.columns)

# print(df['genres'])
# print(df['release_date'])


df['revenue'].fillna(value=0)
df['release_date'].fillna(value=0)
df['budget'].fillna(value=0)
df = df.dropna(subset=['genres'])
print(df['release_date'].head())
df['release_date'] = pd.to_datetime(df['release_date']).dt.year

df.loc[df['release_date'] > 2017, 'release_date'] = df['release_date'] - 100

indexBudget = df[df['budget'] < 10000000].index
df.drop(indexBudget, inplace=True)

indexRevenue = df[df['revenue'] < 100000000].index
df.drop(indexRevenue, inplace=True)

# for row in df['genres']:
    # print(row)

df['genres'] = df['genres'].apply(lambda x: list(map(lambda d: list(d.values())[1], ast.literal_eval(x)) if isinstance(x, str) else []))
df['spoken_languages'] = df['spoken_languages'].apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))

print(df.head().genres)
print(df.head().spoken_languages)
# df['genres'] = ast.literal_eval(df['genres'])

# Access the first value in the list of each row.
# [Drama, Comedy] = [Drama]
# [Thriller, Drama, Comedy] = [Thriller]
df['genres'] = df['genres'].apply(lambda x: x[0])



# print(df['runtime'])
# print(df['genres'])

sns.pairplot(x_vars="release_date", y_vars="revenue", hue='genres', data=df)
sns.pairplot(x_vars="release_date", y_vars="budget", data=df, hue="genres")

# sns.pairplot(df)
plt.show()





