from sklearn.tree import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


import ast
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load in the dataset using pandas
df = pd.read_csv('train.csv')



# summary of the first values in all the columns to get an idea of how they are formatted
# for value in df.columns:
#     print(value)
#     print(df[value].head())
# print(df.head())
# print(df.columns.tolist())
print(df['revenue'].describe())
# #print(df.corr())

# LabelEncoder allows us to convert our categorical values to a numerical output for use in our model
number = LabelEncoder()

#  creating a series for our genres
genre = df['genres']


# create an empty list that will be added to our data frame once populated
genres = []
# loop through the genres column and get every genre
for row in genre:
    # Create an empty list to add the various genres for each row
    row_genre = []
    if type(row) is str:
        # Convert every row in the list to be a list of dictionaries
        row = ast.literal_eval(row)
        for entry in row:
            row_genre.append(entry['name'])
        row = row_genre
    else:
        row_genre.append("none")
    # Add the row genre to the list of genres for the data frame
    genres.append(row_genre)


# extract the first genres for every row
first_genre = []
for row in genres:
    row = row[0]
    first_genre.append(row)


# create a series for our production companies
prod_company = df['production_companies']
# create an empty list that will be added to our data frame once populated
prod_companies = []
# loop through the prod companies column and get every company
for row in prod_company:
    # Create an empty list to add the various companies for each row
    row_company = []
    if type(row) is str:
        # Convert every row in the list to be a list of dictionaries
        row = ast.literal_eval(row)
        for entry in row:
            row_company.append(entry['id'])
        row = row_company
    else:
        row_company.append("none")
    # Add the row company to the list of companies for the data frame
    prod_companies.append(row_company)


first_company = []
for row in prod_companies:
    row = row[0]
    first_company.append(row)


df['first_company'] = first_company
df['first_genre'] = first_genre






# create our features to use in our model
features = ['budget', 'runtime', 'first_genre', 'original_language', 'first_company']
# transform our data and fill in any blank values
df['runtime'].fillna(df['runtime'].mean(), inplace=True)
df['budget'].replace(0, df['budget'].mean(), inplace=True)
# df['first_genre'] = number.fit_transform(df['first_genre'].astype('str'))
df['original_language'] = number.fit_transform(df['original_language'].astype('str'))
# df['first_company'] = number.fit_transform(df['first_company'].astype('str'))

# print(df[features].head())
X = df[features]
y = df.revenue
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


sns.set(style="whitegrid")

sns.scatterplot(x="budget", y="revenue", data=df)
plt.show()



