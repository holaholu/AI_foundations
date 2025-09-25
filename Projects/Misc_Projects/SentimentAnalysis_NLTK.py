#NLTK mean Natural Language Toolkit. It is a library for working with human language data.

import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
import random

#Download NLTK data files. Run only once and comment out after
# nltk.download("movie_reviews")
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("punkt_tab")

#Preprocess the data and extract features
def extract_features(words):
    return {word: True for word in words}

#Load the movie reviews dataset from NLTK
documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)] #this is a list of tuples where each tuple contains a list of words and the category (positive or negative)

#Shuffle the documents to ensure randomization
random.shuffle(documents)

#Prepare the dataset for training and testing
featuresets = [(extract_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[:1600], featuresets[1600:]

#Train the Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

#Evaluate the classifier
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy of the classifier: {accuracy * 100}%")

#Show the most informative features
classifier.show_most_informative_features(10)

#Test on new input sentences
def analyze_sentiment(text):
    #Tokenize the input text
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words("english")]
    #Predict the sentiment
    features = extract_features(words)
    #Classify the input text
    return classifier.classify(features)

#Test the classifier with some custom text inputs
test_sentences = [
    "I love this movie! It's amazing!",
    "I hate this movie! It's terrible!",
    "This movie is so good!",
    "This movie is so bad!",
    "I have mixed feelings about this movie"
  ]

for sentence in test_sentences:
    print("\n")
    print("Sentence: ", sentence)
    print("Predicted Sentiment: ", analyze_sentiment(sentence))

    







