from transformers import pipeline
import re

# Load a chatbot pipeline
chatbot_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium")

#initialize memory storage for conversation
memory = {}

def chat_with_memory(user_input):
    #check if the user mentions their name
    lower_text = user_input.lower()
    if "my name is" in lower_text:
        # Find the index in the original string corresponding to the lowercase match
        idx = lower_text.find("my name is")
        name_raw = user_input[idx + len("my name is"):].strip()
        # Clean the extracted name: stop at first punctuation or excess text
        name = re.split(r"[,.!?:;\n]", name_raw, maxsplit=1)[0].strip()
        # Fallback if empty
        if not name:
            name = "there"
        memory["name"] = name
        return f"It's nice to meet you, {name}!"
    
    #use memory in responses
    if "name" in memory:
        response = chatbot_pipeline(f"Hello {memory['name']}, how can I help you today?") 
    else:
        response = chatbot_pipeline(user_input)    
    return response[0]['generated_text']

# Create a main function
def main():
    print("Welcome to the Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chat_with_memory(user_input)
        print("Chatbot: ", response)

# Run the main function
main()


        
