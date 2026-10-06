## Imports
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

import streamlit as st

## Env
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path = env_path)

## LLM
llm = ChatGroq(model = "openai/gpt-oss-20b", streaming=True)

## Tool - Google Search Tool
search = GoogleSerperAPIWrapper()
tools = [search.run]

## Memory
if "memory" not in st.session_state:
	st.session_state.memory = MemorySaver()
	st.session_state.history = []

## Agent
agent = create_agent(
	model = llm,
	tools = tools,
	checkpointer = st.session_state.memory,
	system_prompt = "You are an amazing AI agent which can search on Google as well",
)

st.subheader("QuickAnswer - QnA Bot")
st.markdown("Answers at the speed of your thought")

for message in st.session_state.history:
	role = message["role"]
	content = message["content"]
	st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything ..!!")
if query:
	st.chat_message("user").markdown(query)
	st.session_state.history.append({"role":"user", "content": query})

	response = agent.stream(
		{"messages":[{"role":"user", "content": query}]},
		{"configurable": {"thread_id":"RK1"}},
		stream_mode = "messages"
	)

	ai_container = st.chat_message("ai")
	with ai_container:
		space = st.empty()

		message = ""

		for chunk in response:
			message = message + chunk[0].content
			space.write(message)

		st.session_state.history.append({"role":"ai", "content": message})
