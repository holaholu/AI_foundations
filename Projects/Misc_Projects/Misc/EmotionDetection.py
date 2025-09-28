#import the necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#load the dataset. Comment out the download lines if you have already downloaded the datasets
# nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('vader_lexicon')

#Initialize the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

#Sample text for emotion detection
text ="""
I am so happy today. I feel like I can conquer the world. 
"""

text2 ="""
I feel good and sad at the same time. Not sure if to laugh or cry.
"""

#function to detect emotion
def detect_emotion(text):
    #Analyze sentiment
    scores = sid.polarity_scores(text)#return a dictionary of sentiment scores

    #display the sentiment scores
    print("Sentiment scores:", scores)
    
    #Determine the emotion
    if scores['compound'] >= 0.5:
        emotion = "Joy"
    elif scores['compound'] <= -0.5:
        emotion = "Sadness"
    elif scores["neg"]> 0.5:
        emotion = "Anger"
    elif scores["neu"]> 0.7:
        emotion = "Neutral"
    else:
        emotion = "Mixed Emotions"
    return emotion

#Detect emotion in the sample text
detected_emotion = detect_emotion(text2)
print("Detected emotion:", detected_emotion)


    
        