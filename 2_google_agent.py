from pathlib import Path
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path = env_path)

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
	model = llm,
	tools = [search.run],
	checkpointer = MemorySaver(),
	system_prompt = "You are a agent and can search for any question on google.",
)

while True:
	query = input("User: ")
	if query.lower() == "quit":
		print("AI: Goodbye")
		break

	response = agent.invoke(
		{"messages":[{"role":"user", "content":query}]},
		{"configurable": {"thread_id":"RK1"}},
	)
	print("AI: ", response["messages"][-1].content)
