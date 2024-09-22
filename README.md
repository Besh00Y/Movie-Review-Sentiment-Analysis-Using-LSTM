# Sentiment Analysis using IMDB Dataset of 50k Movie Reviews

This project focuses on building a deep learning model to classify movie reviews into positive or negative sentiments using the IMDB dataset. The model is built using Python, TensorFlow, and Natural Language Processing (NLP) techniques, and it is deployed via a user-friendly interface created using Gradio.

##Dataset
The dataset used is the IMDB Dataset of 50K Movie Reviews, available on Kaggle. This dataset contains 50,000 movie reviews, each labeled as either positive or negative.

## Download the dataset:
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Project Workflow
Data Preprocessing:

The reviews are cleaned by removing HTML tags, URLs, and non-alphabetic characters.
Tokenization and stemming (using NLTK) are applied to prepare the reviews for model training.
Stopwords are removed to focus on meaningful words in the reviews.
Model Creation:

A sequential model is created using TensorFlow/Keras, consisting of an embedding layer, an LSTM layer, and a dense output layer.
The model is trained using binary cross-entropy loss and Adam optimizer for 10 epochs.
Training and Evaluation:

The dataset is split into 80% training and 20% testing.
The model is evaluated on the test set with metrics like accuracy and loss.
Deployment with Gradio:

A simple user interface is built using Gradio, where users can input movie reviews, and the trained model will predict the sentiment along with the confidence score.

## Installation and Setup
Requirements
  Python 3.x
  TensorFlow
  NLTK
  NumPy
  Pandas
  Scikit-learn
  Matplotlib
  Seaborn
  Gradio
  Kaggle API (to download the dataset)
  Instructions

## Install dependencies:
pip install -r requirements.txt

## Download and unzip the dataset:
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
!unzip imdb-dataset-of-50k-movie-reviews.zip

Run the training script: The model can be trained using the script provided. After training, it will save the model as sentiment_model.h5.
python train_model.py
Run the Gradio app: Launch the Gradio interface to test the model:
python app.py


Here’s a sample README file for your movie review sentiment analysis project:

Movie Review Sentiment Analysis
This project focuses on building a deep learning model to classify movie reviews into positive or negative sentiments using the IMDB dataset. The model is built using Python, TensorFlow, and Natural Language Processing (NLP) techniques, and it is deployed via a user-friendly interface created using Gradio.

Dataset
The dataset used is the IMDB Dataset of 50K Movie Reviews, available on Kaggle. This dataset contains 50,000 movie reviews, each labeled as either positive or negative.

##Download the dataset:
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Project Workflow
Data Preprocessing:

The reviews are cleaned by removing HTML tags, URLs, and non-alphabetic characters.
Tokenization and stemming (using NLTK) are applied to prepare the reviews for model training.
Stopwords are removed to focus on meaningful words in the reviews.
Model Creation:

A sequential model is created using TensorFlow/Keras, consisting of an embedding layer, an LSTM layer, and a dense output layer.
The model is trained using binary cross-entropy loss and Adam optimizer for 10 epochs.
Training and Evaluation:

The dataset is split into 80% training and 20% testing.
The model is evaluated on the test set with metrics like accuracy and loss.
Deployment with Gradio:

A simple user interface is built using Gradio, where users can input movie reviews, and the trained model will predict the sentiment along with the confidence score.
Installation and Setup
Requirements
Python 3.x
TensorFlow
NLTK
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Gradio
Kaggle API (to download the dataset)
Instructions

## Install dependencies:
pip install -r requirements.txt

## Download and unzip the dataset:
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
!unzip imdb-dataset-of-50k-movie-reviews.zip
Run the training script: The model can be trained using the script provided. After training, it will save the model as sentiment_model.h5.
python train_model.py

## Run the Gradio app: Launch the Gradio interface to test the model:
python app.py
Usage
Enter a movie review in the input textbox and click the Analyze button. The model will predict whether the review is positive or negative and display the confidence score.

## Example:

Input: I absolutely loved this movie. It was fantastic!
Output: The review is positive with 98.74% confidence.
Model Architecture
Embedding Layer: Converts the text input into dense word vectors.
LSTM Layer: Processes the word vectors and captures sequential dependencies.
Dense Output Layer: Produces a binary output, indicating whether the sentiment is positive or negative.
Results
The model achieves around 87% accuracy on the test set, which is quite effective for binary sentiment classification tasks.

## Evaluation Metrics:
Accuracy: Measures how well the model predicts the correct sentiment.
Loss: Tracks the error during training and testing.
Gradio Interface

The Gradio interface allows real-time predictions for movie reviews. Just input the text, and get the sentiment prediction.

## Future Work
Improving the accuracy by tuning hyperparameters or using more advanced techniques like Bidirectional LSTMs or transformer-based models like BERT.
Expanding the dataset to include more diverse reviews.

## Conclusion
This project demonstrates a simple yet effective approach to movie review sentiment analysis using deep learning. The deployment through Gradio makes it user-friendly, allowing non-technical users to interact with the model easily.
