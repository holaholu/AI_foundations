import re #regular expression module for pattern matching

#A dictionary that maps keywords to responses
responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hi there! How can I help you today?",
    "how are you": "I'm just a computer program, so I don't have feelings, but thanks for asking!",
    "what is your name": "I'm a simple chatbot created to assist you with your tasks.",
    "help": "I can help you with a variety of tasks, such as answering questions, providing information, and more.",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're welcome!",
    "default": "I'm sorry, I didn't understand that. Can you please rephrase?"
}

#Function to process user input and generate a response
def chatbot_response(user_input):
    #Convert user input to lowercase
    user_input = user_input.lower()
    
    for keyword in responses:
        if re.search(keyword, user_input):
            return responses[keyword]
    
    return responses["default"]

#Main loop to run the chatbot
def chatbot():
    print("Chatbot: Hello! I'm here to help you. (Type 'bye' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot_response(user_input)
        print("Chatbot: " + response)

#Run the chatbot
if __name__ == "__main__":
    chatbot()
