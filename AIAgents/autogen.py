
from autogen import AssistantAgent, UserProxyAgent 

#import API key from environment variable
import os
API_KEY = os.getenv("OPENAI_API_KEY")

llm_config = {"model":"gpt-4o-mini", "api_key": API_KEY}

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy", code_execution_config=False) #code_execution_config=False means the user cannot execute code

user_proxy.initiate_chat(assistant, message="Tell me a joke about tech stocks.")
