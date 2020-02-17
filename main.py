import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
from sklearn import metrics
from sklearn.feature_selection import RFE


data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


def clean_string_literal(string_series, dic_value, new_col_name, data):
    row_values = []
    # loop through the genres column and get every genre
    for row in string_series:
        # Create an empty list to add the various genres for each row
        dic_values = []
        if type(row) is str:
            # Convert every row in the list to be a list of dictionaries
            dic = ast.literal_eval(row)
            for entry in dic:
                dic_values.append(entry[str(dic_value)])
            dic = dic_values
        else:
            dic_values.append("none")
        # Add the row genre to the list of genres for the data frame
        row_values.append(dic_values)

    # extract the first genres for every row
    first_val = []
    for row in row_values:
        row = row[0]
        first_val.append(row)
    data[new_col_name] = first_val


clean_string_literal(string_series=data['genres'], dic_value="name", new_col_name='first_genre', data=data)
clean_string_literal(string_series=data['production_companies'], dic_value="name", new_col_name='first_company',data=data)
data['runtime'].fillna(data['runtime'].mean(), inplace=True)

clean_string_literal(string_series=test_data['genres'], dic_value="name", new_col_name='first_genre', data=test_data)
clean_string_literal(string_series=test_data['production_companies'], dic_value="name", new_col_name='first_company', data=test_data)
test_data['runtime'].fillna(test_data['runtime'].mean(), inplace=True)

features = ['budget', 'runtime', 'popularity', 'first_genre', 'original_language',  'first_company']

# print(df[features].head())
X = data[features]
X = pd.get_dummies(data=X, columns=['first_genre', 'original_language', 'first_company'])
test_X = test_data[features]
test_X = pd.get_dummies(data=test_X, columns=['first_genre', 'original_language', 'first_company'])
final_train, final_test = X.align(test_X, join='inner', axis=1)
y = data.revenue
# train_X, val_X, train_y, val_y = train_test_split(final_train, y, random_state=1)

# train our model and make predictions
imdb_test_model = RandomForestRegressor(n_estimators=240, random_state=1, max_leaf_nodes=1300)
imdb_test_model.fit(final_train, y)
# val_predictions = imdb_test_model.predict(final_train)


# bring in our test data, clean it and apply our model

test_preds = imdb_test_model.predict(final_test)

output = pd.DataFrame({'Id': test_data.id,
                       'revenue': test_preds})
output.to_csv('submission.csv', index=False)


