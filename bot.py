import os
from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
load_dotenv()

import os

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")  # 🔒 Replace with your actual key



class AgentState(TypedDict):
    messages : List[HumanMessage]
    
llm =  ChatGoogleGenerativeAI(model="gemini-2.5-pro")

def process(state:AgentState)->AgentState:
    response = llm.invoke(state["messages"])
    print(response.content)

    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

user_input = input("Enter :")
while user_input != "exit":
   agent.invoke({"messages" : [HumanMessage(content=user_input)]})
   user_input = input("Enter :")
   
