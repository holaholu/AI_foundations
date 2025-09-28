#import the required libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

#Download NLTK data files. Run only once and comment out after
#nltk.download("punkt")
# nltk.download("stopwords")

#example text for summarization
text = """
Artificial Intelligence (AI) is a field of computer science that focuses on creating intelligent machines that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and natural language processing. AI has the potential to transform many aspects of our lives, from healthcare and transportation to education and entertainment. However, it also raises important ethical and societal concerns, such as job displacement, privacy, and bias. As AI continues to advance, it is important to consider its impact on society and work towards responsible and ethical AI development and deployment.
"""

#Function to generate a frequency-based summary
def summarize_text(text, num_sentences=2):
    #Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    #Remove stopwords and punctuation/non-alphanumeric characters
    stop_words = set(stopwords.words("english"))
    word_frequencies = {}
    for word in words:
        if word.isalpha() and word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
    #Score each sentence based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]
    
#Sort sentences by score and select top `num_sentences`
    summary_sentences = sorted(sentence_scores,key=sentence_scores.get,reverse=True)[:num_sentences]
    summary = " ".join(summary_sentences)
    return summary

#Generate summary
summary = summarize_text(text, num_sentences=2)
print("Original Text:")
print(text)
print("\n")
print("Summary:")
print(summary)

