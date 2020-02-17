import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
from matplotlib import interactive
import seaborn as sns

# print(os.listdir(os.getcwd()))
# Preview the files that are in the project directory

data = pd.read_csv('train.csv')
# Load in the csv file using pandas

df = pd.DataFrame(data)
# convert to a pandas data frame

print(df.head())
# Preview Data

print(df.columns)
# Find column names

sns.set()

genreAvgRevenue = pd.DataFrame(data[['revenue', 'genres']])
# create data frame from revenue and genre
# print(genreAvgRevenue.head())
# print(genreAvgRevenue['genres'].tail())

# create an empty list that will be added to our data frame once populated
genres = []
# loop through the genres column and get every genre
for row in genreAvgRevenue['genres']:
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

# add our genres list to the data frame genreavgrevenue
genreAvgRevenue['actual_genres'] = genres

# print columns to make sure our list was added
print(sorted(genreAvgRevenue))

# print(list(genreAvgRevenue['actual_genres'].values.tolist()))


# extract the first genres for every row
first_genre = []
for row in genreAvgRevenue['actual_genres']:
    row = row[0]
    first_genre.append(row)

# add first genres to data frame
genreAvgRevenue['first_genre'] = first_genre

print(list(genreAvgRevenue.columns.tolist()))
# create a list of distinct genres so that we can graph it
distinct_genres = ['Drama', 'Family', 'Science Fiction', 'Music', 'Thriller', 'TV Movie', 'None', 'Documentary',
                   'History', 'Horror', 'War', 'Crime', 'Western', 'Adventure', 'Foreign', 'Comedy', 'Mystery',
                   'Fantasy', 'Action', 'Animation', 'Romance']

# want to take the set of first genres and sum the revenue and plot

print(distinct_genres)

drama_revenue = []
family_revenue = []
scienceFiction_revenue = []
music_revenue = []
thriller_revenue = []
tvMovie_revenue = []
none_revenue = []
documentary_revenue = []
history_revenue = []
horror_revenue = []
war_revenue = []
crime_revenue = []
western_revenue = []
adventure_revenue = []
foreign_revenue = []
comedy_revenue = []
mystery_revenue = []
fantasy_revenue = []
action_revenue = []
animation_revenue = []
romance_revenue = []

for index, row in genreAvgRevenue.iterrows():
    if row[3] == 'Drama':
        # print(row['first_genre'])
        drama_revenue.append(row[0])
    elif row[3] == 'Family':
        family_revenue.append(row[0])
    elif row[3] == 'Science Fiction':
        scienceFiction_revenue.append(row[0])
    elif row[3] == 'Music':
        music_revenue.append(row[0])
    elif row[3] == 'Thriller':
        thriller_revenue.append(row[0])
    elif row[3] == 'TV Movie':
        tvMovie_revenue.append(row[0])
    elif row[3] == 'none':
        none_revenue.append(row[0])
    elif row[3] == 'Documentary':
        documentary_revenue.append(row[0])
    elif row[3] == 'History':
        history_revenue.append(row[0])
    elif row[3] == 'Horror':
        horror_revenue.append(row[0])
    elif row[3] == 'War':
        war_revenue.append(row[0])
    elif row[3] == 'Crime':
        crime_revenue.append(row[0])
    elif row[3] == 'Western':
        western_revenue.append(row[0])
    elif row[3] == 'Adventure':
        adventure_revenue.append(row[0])
    elif row[3] == 'Foreign':
        foreign_revenue.append(row[0])
    elif row[3] == 'Comedy':
        comedy_revenue.append(row[0])
    elif row[3] == 'Mystery':
        mystery_revenue.append(row[0])
    elif row[3] == 'Fantasy':
        fantasy_revenue.append(row[0])
    elif row[3] == 'Action':
        action_revenue.append(row[0])
    elif row[3] == 'Animation':
        animation_revenue.append(row[0])
    elif row[3] == 'Romance':
        romance_revenue.append(row[0])

sum_of_revenue = []
sum_drama_revenue = sum(drama_revenue)
sum_family_revenue = sum(family_revenue)
sum_scienceFiction_revenue = sum(scienceFiction_revenue)
sum_music_revenue = sum(music_revenue)
sum_thriller_revenue = sum(thriller_revenue)
sum_tvMovie_revenue = sum(tvMovie_revenue)
sum_none_revenue = sum(none_revenue)
sum_documentary_revenue = sum(documentary_revenue)
sum_history_revenue = sum(history_revenue)
sum_horror_revenue = sum(horror_revenue)
sum_war_revenue = sum(war_revenue)
sum_crime_revenue = sum(crime_revenue)
sum_western_revenue = sum(western_revenue)
sum_adventure_revenue = sum(adventure_revenue)
sum_foreign_revenue = sum(foreign_revenue)
sum_comedy_revenue = sum(comedy_revenue)
sum_mystery_revenue = sum(mystery_revenue)
sum_fantasy_revenue = sum(fantasy_revenue)
sum_action_revenue = sum(action_revenue)
sum_animation_revenue = sum(animation_revenue)
sum_romance_revenue = sum(romance_revenue)

sum_of_revenue.append(sum_drama_revenue)
sum_of_revenue.append(sum_family_revenue)
sum_of_revenue.append(sum_scienceFiction_revenue)
sum_of_revenue.append(sum_music_revenue)
sum_of_revenue.append(sum_thriller_revenue)
sum_of_revenue.append(sum_tvMovie_revenue)
sum_of_revenue.append(sum_none_revenue)
sum_of_revenue.append(sum_documentary_revenue)
sum_of_revenue.append(sum_history_revenue)
sum_of_revenue.append(sum_horror_revenue)
sum_of_revenue.append(sum_war_revenue)
sum_of_revenue.append(sum_crime_revenue)
sum_of_revenue.append(sum_western_revenue)
sum_of_revenue.append(sum_adventure_revenue)
sum_of_revenue.append(sum_foreign_revenue)
sum_of_revenue.append(sum_comedy_revenue)
sum_of_revenue.append(sum_mystery_revenue)
sum_of_revenue.append(sum_fantasy_revenue)
sum_of_revenue.append(sum_action_revenue)
sum_of_revenue.append(sum_animation_revenue)
sum_of_revenue.append(sum_romance_revenue)

# print(sum_of_revenue)

y_pos = np.arange(len(distinct_genres))
# create a bar graph of the sum of the revenue for each genre
# plt.bar(x=y_pos, height=sum_of_revenue, width=.5)
# plt.xticks(y_pos, distinct_genres, fontsize=9, rotation=45)
# plt.ylabel('Sum of Revenue')
# plt.title('Sum of Revenue by Genre')
# plt.show()


average_of_revenue = []
avg_drama_revenue = sum(drama_revenue) / len(drama_revenue)
avg_family_revenue = sum(family_revenue) / len(family_revenue)
avg_scienceFiction_revenue = sum(scienceFiction_revenue) / len(scienceFiction_revenue)
avg_music_revenue = sum(music_revenue) / len(music_revenue)
avg_thriller_revenue = sum(thriller_revenue) / len(thriller_revenue)
avg_tvMovie_revenue = sum(tvMovie_revenue) / len(tvMovie_revenue)
avg_none_revenue = sum(none_revenue) / len(none_revenue)
avg_documentary_revenue = sum(documentary_revenue) / len(documentary_revenue)
avg_history_revenue = sum(history_revenue) / len(history_revenue)
avg_horror_revenue = sum(horror_revenue) / len(horror_revenue)
avg_war_revenue = sum(war_revenue) / len(war_revenue)
avg_crime_revenue = sum(crime_revenue) / len(crime_revenue)
avg_western_revenue = sum(western_revenue) / len(western_revenue)
avg_adventure_revenue = sum(adventure_revenue) / len(adventure_revenue)
avg_foreign_revenue = sum(foreign_revenue) / len(foreign_revenue)
avg_comedy_revenue = sum(comedy_revenue) / len(comedy_revenue)
avg_mystery_revenue = sum(mystery_revenue) / len(mystery_revenue)
avg_fantasy_revenue = sum(fantasy_revenue) / len(fantasy_revenue)
avg_action_revenue = sum(action_revenue) / len(action_revenue)
avg_animation_revenue = sum(animation_revenue) / len(animation_revenue)
avg_romance_revenue = sum(romance_revenue) / len(romance_revenue)

average_of_revenue.append(avg_drama_revenue)
average_of_revenue.append(avg_family_revenue)
average_of_revenue.append(avg_scienceFiction_revenue)
average_of_revenue.append(avg_music_revenue)
average_of_revenue.append(avg_thriller_revenue)
average_of_revenue.append(avg_tvMovie_revenue)
average_of_revenue.append(avg_none_revenue)
average_of_revenue.append(avg_documentary_revenue)
average_of_revenue.append(avg_history_revenue)
average_of_revenue.append(avg_horror_revenue)
average_of_revenue.append(avg_war_revenue)
average_of_revenue.append(avg_crime_revenue)
average_of_revenue.append(avg_western_revenue)
average_of_revenue.append(avg_adventure_revenue)
average_of_revenue.append(avg_foreign_revenue)
average_of_revenue.append(avg_comedy_revenue)
average_of_revenue.append(avg_mystery_revenue)
average_of_revenue.append(avg_fantasy_revenue)
average_of_revenue.append(avg_action_revenue)
average_of_revenue.append(avg_animation_revenue)
average_of_revenue.append(avg_romance_revenue)

# create a bar graph of the avg of revenue by genres
# plt.bar(x=y_pos, height=average_of_revenue,width=.5)
# plt.xticks(y_pos, distinct_genres, fontsize=9, rotation=45)
# plt.ylabel('Avg of Revenue')
# plt.title('Avg of Revenue by Genre')
# plt.show()

# overlay the bar graph of summed revenued with line graph of the average
# plt.figure()
# plt.bar(y_pos, sum_of_revenue, width=.75)
# plt.xticks(y_pos, distinct_genres, fontsize=9, rotation=45)
# plt.ylabel('Sum of Revenue')
# axes2 = plt.twinx()
# axes2.plot(y_pos, average_of_revenue)
# axes2.set_ylabel('Avg of Revenue')
# plt.show()

genre_summary_stats = pd.DataFrame({'genres': distinct_genres,
                                    'sum of revenue': sum_of_revenue,
                                    'average revenue': average_of_revenue})

# print(genre_summary_stats.head())

genreAvgBudget = df[['genres', 'budget']]

# print(genreAvgBudget.head())

sns.relplot(x="genres", y="average revenue", data=genre_summary_stats)
plt.show()
