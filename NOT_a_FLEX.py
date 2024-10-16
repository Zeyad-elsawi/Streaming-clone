import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python Not_a_FLEX.py <movies_file> <ratings_file>")
    else:
        user_scaled, movies_scaled, movies_original = load_data(sys.argv[1], sys.argv[2])

        # Splitting user data
        X_train, X_test = train_test_split(user_scaled, test_size=0.4, random_state=42)
        # Movies data is used for both training and testing
        y_train, y_test = movies_scaled[:len(X_train)], movies_scaled[len(X_train):]

        model = train_model(X_train, y_train, X_test, y_test)

        # Get user input data
        user_data = get_user_data()

        # Recommend movie based on trained model
        recommend_movie(model, user_data, movies_scaled, movies_original)


import pandas as pd


def load_data(movies_file, ratings_file):
    # Load the data from CSV files
    movies = pd.read_csv(movies_file)
    ratings = pd.read_csv(ratings_file)

    # Print the columns to debug
    print("Movies DataFrame columns:", movies.columns)

    # Check if 'title' exists in movies DataFrame
    if 'title' not in movies.columns:
        raise KeyError("The 'title' column is missing in the movies DataFrame.")

    # Create a list of genres for binary encoding
    genres = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller'
    ]

    # Prepare the movies DataFrame
    # Extract year from title and compute average ratings
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

    # Calculate average ratings for each movie
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings.columns = ['movieId', 'avg_rating']

    # Merge the average ratings with movies DataFrame
    movies = movies.merge(avg_ratings, on='movieId', how='left')

    # One-hot encode genres into binary columns
    for genre in genres:
        movies[genre] = movies['genres'].str.contains(genre).astype(int)

    # Drop the original genres column
    movies = movies.drop(columns=['genres', 'title'], axis=1)

    # Prepare the user DataFrame
    user_data = ratings.groupby('userId').agg(
        rating_count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()

    # Calculate average ratings for each genre per user
    for genre in genres:
        user_data[genre + '_avg'] = \
        ratings[ratings['movieId'].isin(movies[movies[genre] == 1]['movieId'])].groupby('userId')[
            'rating'].mean().fillna(0)

    # Merge user data with the binary genre data
    user_data = user_data.fillna(0)
    # Return user data, movies, and the original movies DataFrame
    return user_data, movies, movies



def train_model(X_train, y_train, X_test, y_test):
    # Neural networks for user and movie vectors
    user_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32),
    ])

    movie_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32),
    ])

    # Inputs for users and movies
    user_input = tf.keras.Input(shape=(X_train.shape[1],))
    movie_input = tf.keras.Input(shape=(y_train.shape[1],))

    # Get user and movie vectors
    user_vector = user_NN(user_input)
    movie_vector = movie_NN(movie_input)


    # Compute dot product
    output = tf.keras.layers.Dot(axes=1)([user_vector, movie_vector])

    # Build and compile the model
    model = tf.keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.MeanSquaredError())

    # Train the model
    model.fit([X_train, y_train], y_train, epochs=10, batch_size=32)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    return model


def get_user_data():
    ratings_df = pd.read_csv('ratings.csv')  # Adjust the filename as needed
    lastid = ratings_df['userId'].iloc[-1]
    user_data = []

    # Collecting user input for the ratings
    user_data.append(int(lastid + 1))
    user_data.append(float(input("Enter the Average rating: ")))
    user_data.append(float(input("Enter the rating of Action: ")))
    user_data.append(float(input("Enter the rating of Adventure: ")))
    user_data.append(float(input("Enter the rating of Animation: ")))
    user_data.append(float(input("Enter the rating of Children: ")))
    user_data.append(float(input("Enter the rating of Comedy: ")))
    user_data.append(float(input("Enter the rating of Crime: ")))
    user_data.append(float(input("Enter the rating of Documentary: ")))
    user_data.append(float(input("Enter the rating of Drama: ")))
    user_data.append(float(input("Enter the rating of Fantasy: ")))
    user_data.append(float(input("Enter the rating of Horror: ")))
    user_data.append(float(input("Enter the rating of Mystery: ")))
    user_data.append(float(input("Enter the rating of Romance: ")))

    return user_data


def recommend_movie(model, user_data, movies_scaled, movies_original):
    user_vector = np.expand_dims(user_data, axis=0)
    predicted_ratings = model.predict([user_vector, movies_scaled])

    # Get the index of the highest predicted rating
    top_movie_index = np.argmax(predicted_ratings)

    # Retrieve the recommended movie from the original dataset
    recommended_movie = movies_original.iloc[top_movie_index]
    print(f"We recommend you to watch: {recommended_movie['title']}")


if __name__ == "__main__":
    main()
