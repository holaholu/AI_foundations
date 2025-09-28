#import required libraries
import pandas as pd
import numpy as np
from surprise import Dataset,Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split,cross_validate
import seaborn as sns
import matplotlib.pyplot as plt

#Load the dataset
df  = pd.read_csv('./data/ratings.csv')
df.drop('timestamp',axis=1,inplace=True)
print(df.head())

#Define a reader object for surprise
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df[['userId','movieId','rating']],reader)

#Split the data into training and testing sets
trainset,testset = train_test_split(data,test_size=0.25,random_state=42)

#Build collaborative filtering model using SVD
model = SVD()
model.fit(trainset)

#Make predictions on the test set
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse:.4f}")

#Make a prediction for a specific user and movie
user_id = 196
movie_id = 242
predicted_rating = model.predict(user_id,movie_id).est
print(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating:.2f}")

#Visualize Distribution of Ratings
plt.figure(figsize=(10,6))
sns.histplot(df['rating'],kde=True,bins=5)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


