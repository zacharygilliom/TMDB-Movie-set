# TMDB-Movie-set

Using the Kaggle dataset, we want to try to predict how much revenue a movie will make

In our final model we decided to go with a Random Forest Regressor.  This was fairly simple to implement.  

Some of the difficulties was in the data cleaning.  Some of our features were imbedded within strings in the cells.  For example our Genre and Production Companies were a bit tough to get, but we ended up being able to isolate the first genre for each movie and the production company.

As this was a competition, our output is a csv file named submission.csv.

On our way to the submission, we want to do some EDA.  I created some scatterplots to help show the relationship with some of our features.

![budgetvreleasedate](https://user-images.githubusercontent.com/23482152/74619787-bf559f00-5104-11ea-89a3-34dcb5218872.png)

This shows how our budget relates to the release date.  As expected we see that movies have gone up in budget as time goes on.

![revenuevreleasedate](https://user-images.githubusercontent.com/23482152/74619817-db594080-5104-11ea-9f4c-d4e6bb736481.png)

Taking a look at the revenue, we see the same relationship.

One of the more interesting pieces is seeing how the genre relates to the budget and revenue.  It tends to show that Action and Drama movies have more outliers in terms of budgets and revenues.