Movie Recommendation System
Overview
This project is a movie recommendation system that predicts the best movie for a user based on their genre preferences and ratings. It uses a neural network model trained on the MovieLens dataset.

Features
Loads and processes MovieLens dataset (movies and ratings)
One-hot encodes movie genres for feature extraction
Creates user profiles based on their rating history
Trains a neural network to predict user preferences
Recommends movies based on the highest predicted rating
Installation
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
Install dependencies:
bash
Copy
Edit
pip install numpy pandas tensorflow scikit-learn
Usage
Ensure the dataset files movies.csv and ratings.csv are in the same directory.
Run the script to train the model:
bash
Copy
Edit
python train.py
Get movie recommendations:
python
Copy
Edit
from recommend import recommend_movie
recommended_movie = recommend_movie(model, user_data, movies_scaled, movies_original)
print(recommended_movie)
Project Structure
perl
Copy
Edit
📁 movie-recommendation-system
 ├── movies.csv            # Movie metadata
 ├── ratings.csv           # User ratings dataset
 ├── train.py              # Model training script
 ├── recommend.py          # Movie recommendation logic
 ├── utils.py              # Data processing functions
 ├── README.md             # Project documentation
Model Details
User and movie embeddings are learned using a deep neural network
Input: User preferences & movie attributes
Output: Predicted rating for each movie
