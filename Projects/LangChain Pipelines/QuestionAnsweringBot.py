from transformers import pipeline


# Set up a knowledge base
knowledge_base = """
LangChain is a framework for building language models. It is used to build language models for a variety of tasks, including text generation, question answering, and chatbots. It simplifies the process of building language models by providing a high-level API for building and training language models.
"""

# Load a question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define a function to answer questions
def answer_question(question, context = knowledge_base):
    response = qa_pipeline(question=question, context=context)
    answer = response["answer"]
    confidence = response["score"]
    return answer, confidence
    


def main():
    print("Welcome to the Simple Question Answering Bot!")
 
    while True:
        question = input("Ask a question (or 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        answer, confidence = answer_question(question)
        print(f"Bot: {answer}")
        print(f"Confidence: {confidence:.2f}")

main()

    