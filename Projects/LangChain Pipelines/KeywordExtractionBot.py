from transformers import pipeline

#load a keyword extraction pipeline
keyword_pipeline = pipeline("ner", model="ml6team/keyphrase-extraction-distilbert-inspec")

# Define a function to extract keywords
def extract_keywords(text):
    keywords = keyword_pipeline(text)
    extracted_keywords = [keyword["word"] for keyword in keywords]
    return extracted_keywords

# create a main function
def main():
    print("Welcome to the Keyword Extraction Bot!")
    while True:
        text = input("Enter text (or 'exit' to quit): ")
        if text.lower() == "exit":
            print("Goodbye!")
            break
        keywords = extract_keywords(text)
        print("Keywords:", keywords)    

# run the main function
main()
