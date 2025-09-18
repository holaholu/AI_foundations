from transformers import pipeline

# Load a sentiment analysis pipeline
sentiment_pipeline = pipeline( "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    response = sentiment_pipeline(text)
    label = response[0]['label']
    confidence = response[0]['score']
    return label, confidence


def main():
    print("Welcome to the Sentiment Analysis Bot!")
    while True:
        text = input("Enter text (or 'exit' to quit): ")
        if text.lower() == "exit":
            print("Goodbye!")
            break
        label, confidence = analyze_sentiment(text)
        print(f"Sentiment: {label}")
        print(f"Confidence: {confidence:.2f}")

main()
    
