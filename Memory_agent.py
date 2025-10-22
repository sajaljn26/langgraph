import os 
from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")  # 🔒 Replace with your actual key

class AgentState(TypedDict):
    messages : List[Union[HumanMessage,AIMessage]]
    
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def process(state:AgentState)->AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\n{response.content}\n")
    print("Current State : ",state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

conversational_history = []

user_input = input("Enter :")
while user_input != "exit":
   conversational_history.append(HumanMessage(content=user_input))
   result = agent.invoke({"messages":conversational_history})
   conversational_history = result["messages"]
   user_input = input("Enter :")   

with open("logging.txt","w",encoding="utf-8") as file:
    file.write("Your Conversational Log :\n")

    for message in conversational_history:
        if isinstance(message,HumanMessage):
            file.write(f"User : {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"Agent : {message.content}\n")
    file.write("End of Conversational")

print("Conversational Log saved to logging.txt")