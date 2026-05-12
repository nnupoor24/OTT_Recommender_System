import pandas as pd

ratings = pd.read_csv("data/ratings.csv")

user_behavior = ratings.groupby("userId").agg({
    "rating": ["mean", "count"]
})

print(user_behavior.head())