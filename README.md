# Project 5 - Spring 2022

Applied ML Spring 2022 Project 5: NLP, Bag of Words and TF-IDF
Eshan Kumar, ek3227

This project consists of one part.

# Part 1 - Comparing BOW with TF-IDF
In this part, we will train a supervised learning model to predict if a tweet has a positive or negative sentiment. We preprocess text data, then train a Logistic Regression model on the BOW representation of the data, and then a TF-IDF representation of the data, and compare the results on the test data.

## Data Loading
To prepare the data for the supervised learning model, we first create labels for the data, with 1 indicating a tweet with positive sentiment and 0 indicating negative sentiment. Then, we combine the negative and positive data, and split the data into development and test sets. 

## Data Preprocessing
I first remove the '#' symbol from all strings in the data. Then I remove hyperlinks in the tweets using regex. I also use sklearn's ENGLISH_STOP_WORDS, which is a set of words that have been determined to contribute little to the meaning of a string, and remove these stop words from the data. Then, we remove all remaining punctuation from the data using regex. Finally, we apply stemming on the tweets using the NLTK's PorterStemmer(). 

## Model Training and Evaluation
I create bag of words features for each tweet in the development dataset, and then train a logistic regression model on these vectors. Then, I create TF-IDF features for each tweet in the development dataset, and train a logistic regression model on these vectors as well. Finally, I evaluate the performance of these models on the test set, and report the scores. I find that both models have a similar performance on the test dataset, with a test score of ~0.755.
