from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error
import torch

set_verbosity_error()

# Check if MPS (Apple Silicon) or CUDA is available, otherwise use CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = 0  # Use first GPU
else:
    device = -1  # Use CPU

print(f"Using device: {device}")

# Use smaller, more reliable models
summarization_pipeline = pipeline("summarization", 
                                model="facebook/bart-large-cnn", 
                                device=device,
                                max_length=150,
                                min_length=50,
                                truncation=True)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

# Use the same model for refinement to avoid conflicts
refinement_pipeline = pipeline("summarization", 
                             model="facebook/bart-large-cnn", 
                             device=device,
                             max_length=100,
                             min_length=30,
                             truncation=True)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

qa_pipeline = pipeline("question-answering", 
                      model="deepset/roberta-base-squad2", 
                      device=device)

# Simplified approach - summarization models don't need prompt templates
# They work directly with the text input

text_to_summarize = input("\nEnter text to summarize:\n")
length = input("\nEnter the length (short/medium/long): ")

# First summarization
print("\n🔄 Generating initial summary...")
initial_summary = summarization_pipeline(text_to_summarize)
summary_text = initial_summary[0]['summary_text']

# Optional refinement (only if the summary is still long)
if len(summary_text.split()) > 50 and length == "short":
    print("🔄 Refining summary...")
    refined_summary = refinement_pipeline(summary_text)
    summary = refined_summary[0]['summary_text']
else:
    summary = summary_text

print("\n🔹 **Generated Summary:**")
print(summary)

while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
    if question.lower() == "exit":
        break

    try:
        qa_result = qa_pipeline(question=question, context=summary)
        print("\n🔹 **Answer:**")
        print(qa_result["answer"])
        print(f"\n🔹 **Confidence:** {qa_result['score']:.2f}")
    except Exception as e:
        print(f"\n❌ **Error:** Could not answer the question. {str(e)}")