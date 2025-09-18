#text summarization can be extractive or abstractive. Extractive means selecting from the text to create a summary. Abstractive means creating a summary from the text.

from transformers import pipeline

summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    summary = summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False) #do_sample=False means it will not sample from the text
    return summary[0]['summary_text']

def main():
    print("Welcome to the Text Summarization Bot!")
    while True:
        text = input("Enter text (or 'exit' to quit): ")
        if text.lower() == "exit":
            print("Goodbye!")
            break
        summary = summarize_text(text)
        print("Summary:", summary)

main()

    

    