#Setup the environment
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AnyMessage

#Define the agents tools and functions
@tool
def search(query: str):
    """Simulate a web search"""
    if "weather" in query.lower():
        return "It is sunny."
    else:
        return "No data available."

tools = [search]
tool_node = ToolNode(tools)

#Create the Agent Logic
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7).bind_tools(tools)

# Define LangGraph state schema (new API)
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def call_model(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][ -1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

#Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

#add persistence
checkpoint = MemorySaver()
app = workflow.compile(checkpointer=checkpoint)

#Invoke the agent
final_state = app.invoke(
    {"messages": [HumanMessage(content="What is the weather today?")]},
    config={"configurable": {"thread_id": 1}}
     ) 

print(final_state["messages"][ -1].content)
