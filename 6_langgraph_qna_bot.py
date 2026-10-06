from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path = env_path)

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, List

class ChatState(BaseModel):
        messages:Annotated[List, add_messages]

llm = ChatGroq(model="openai/gpt-oss-20b")

def chatBotNode(state:ChatState) -> ChatState:
        res = llm.invoke(state.messages)
        state.messages = [res]
        return state

memory = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chatbot", chatBotNode)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

final_graph = graph.compile(checkpointer = memory)

while True:
	query = input("User :")
	if query.lower() == "quit":
		print("AI : Goodbye")
		break
	res = final_graph.invoke({"messages":[{"role":"user", "content":query}]}, {"configurable":{"thread_id":"1234"}})
	answer = res["messages"][-1].content
	print("AI :", answer)
