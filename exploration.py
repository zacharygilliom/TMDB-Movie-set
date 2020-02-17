from sklearn.tree import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


import ast
import pandas as pd
import numpy as np

# load in the dataset using pandas
df = pd.read_csv('train.csv')
print(df.columns)
print(df.sample(n=5))
print(df['budget'].isin([0]).sum())


# summary of the first values in all the columns to get an idea of how they are formatted
# for value in df.columns:
#     print(value)
#     print(df[value].head())
# print(df.head())
# print(df.columns.tolist())
# print(df['revenue'].describe())
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
            row_company.append(entry['name'])
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

# transform our data and fill in any blank values
df['runtime'].fillna(df['runtime'].mean(), inplace=True)
df['budget'].replace(0, df['budget'].mean(), inplace=True)
# print(df['budget'].isin([0]).sum())
# print(df['revenue'].isin([0]).sum())

features = ['budget', 'runtime', 'popularity', 'first_genre', 'original_language',  'first_company']

# print(df[features].head())
X = df[features]
X = pd.get_dummies(data=X, columns=['first_genre', 'original_language', 'first_company'])
y = df.revenue
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# tmdb_model = RandomForestRegressor(random_state=1)
# n_estimators = [int(x) for x in np.linspace(start=200, stop=450, num=25)]
#
# max_depth = [int(x) for x in np.linspace(0, 300, num=50)]
# max_depth.append(None)
# min_samples_split = [10, 50, 100, 125]
# max_leaf_nodes = [int(x) for x in np.linspace(start=1100, stop=1500, num=25)]
#
# random_grid = {'n_estimators': n_estimators,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'max_leaf_nodes': max_leaf_nodes}
#
# rf_random = RandomizedSearchCV(estimator=tmdb_model, param_distributions=random_grid, n_iter=5, cv=5, verbose=2, n_jobs=-1)
# rf_random.fit(X_train, y_train)
# print(rf_random.best_params_)







# run multiple random forests to find the most ideal number of max leaf nodes
# leaf_nodes = [i for i in range(1300, 1301)]
# estimators = [i for i in range(190, 250, 10)]
# mae_values = []
# for leaf in leaf_nodes:
#     for estimator in estimators:
#         parameter_list = []
#         imdb_test_model = RandomForestRegressor(n_estimators=estimator, random_state=1, max_leaf_nodes=leaf)
#         imdb_test_model.fit(train_X, train_y)
#         val_predictions = imdb_test_model.predict(val_X)
#         val_mae1 = mean_absolute_error(val_predictions, val_y)
#         parameter_list.append(val_mae1)
#         parameter_list.append(leaf)
#         parameter_list.append(estimator)
#         mae_values.append(parameter_list)
#         print(parameter_list)
#
#         # print("MAE for " + str(leaf) + " max leaf nodes and " + str(estimator) + " n_estimators is " + str(val_mae1))
#
# min_mae_value = min([a[0] for a in mae_values])
# print(min_mae_value)

# Model we want to use
imdb_model = RandomForestRegressor(n_estimators=305, random_state=1, max_leaf_nodes=1300, min_samples_split=10, max_depth=293)
imdb_model.fit(X_train, y_train)
val_predictions = imdb_model.predict(X_test)
val_mae1 = mean_absolute_error(val_predictions, y_test)
print("Validation MAE for best value of randomized search CV: {:,.0f}".format(val_mae1))

# imdb_model = LinearRegression()
# imdb_model.fit(train_X, train_y)
# val_predictions = imdb_model.predict(val_X)
# val_mae1 = mean_absolute_error(val_predictions, val_y)
# print(val_mae1)

