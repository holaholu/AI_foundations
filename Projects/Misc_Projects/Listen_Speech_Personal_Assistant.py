#import the required libraries
#Pip install SpeechRecognition
#Pip install pyttsx3, pyaudio
import speech_recognition as sr 
import pyttsx3  #Text to speech library
import webbrowser #for opening websites
import os #for opening files
import datetime #for getting the current time


#initialize the text to speech engine
engine = pyttsx3.init()

#function to make the assistant speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

#function to take a voice command from the user
def take_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            command = recognizer.recognize_google(audio, language='en-in') #uses google's speech recognition to convert speech to text
            print(f"User said: {command}")
        except sr.UnknownValueError:
            print("Sorry, I did not understand that")
            return None
        except sr.RequestError:
            print("Network error. Please try again")
            return None
    return command.lower()

#function to respond to different commands
def respond(command):
    if 'hello' in command or 'hi' in command:
        speak("Hello! How can I assist you today?")
    elif 'search' in command:
        speak("What do you want to search for?")
        search_query = take_command()
        if search_query:
            speak(f"Searching for {search_query}")
            webbrowser.open(f"https://www.google.com/search?q={search_query}")
    elif 'time' in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p") #converts the current time to a string
        speak(f"The current time is {current_time}")
    elif 'open' in command:
        if 'safari' in command:
            speak("Opening Safari")
            os.system("open -a Safari")
        elif 'calculator' in command:
            speak("Opening Calculator")
            os.system("open -a Calculator")
    elif 'bye' in command or 'exit' in command or 'quit' in command:
        speak("Goodbye! Have a great day!")
        exit()
    else:
        speak("Sorry, I did not understand that")

#main function to run the assistant
def run_assistant():
    speak("Hello! I am your personal assistant. How can I assist you today?")
    while True:
        command = take_command()
        if command:
            respond(command)

#run the assistant
if __name__ == "__main__":
    run_assistant()



        
        
