# pip install -r requirements.txt
# huggingface-cli login, then enter token from huggingface.com
"""from huggingface.com, click on models,on left of page, click on libraries, then click on transformers. This list models that work with transformers pipeline. Can also select language/other to filter models. then go back to Tasks. Select task you want to perform. 
Sort models by most downloaded, most stars, most forks, most recent on top right of page. Select a model you like. Click on "use this model" button, then click on transformers.It displays a sample code to use the model and documentation for model. Copy the code and paste it into your project. Some models require you to accept terms and conditions/license before using them. 

Langchain allow you to use multiple models together and make more advanced applications and just deals with LLMs
"""

from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline  # Updated import
from langchain.prompts import PromptTemplate
import torch

# Use a valid and smaller model for testing
model = pipeline("text-generation",
                 model="microsoft/DialoGPT-medium",  # Valid model that exists
                 max_length=256,
                 truncation=True,
                 pad_token_id=50256)  # Add pad token to avoid warnings

llm = HuggingFacePipeline(pipeline=model)  # Use 'pipeline' parameter

# Create a prompt templates
prompt_template = PromptTemplate.from_template(
    "Explain {topic} in detail for a {age} year old to understand")

# Create the chains properlly\
chain = prompt_template | llm

topic = input("Topic: ")
age = input("Age: ")

# Execute chain
response = chain.invoke({"topic": topic, "age": age})
print(response)