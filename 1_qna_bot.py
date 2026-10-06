from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

llm = ChatGoogleGenerativeAI(
      model="gemini-3.7-flash"
)

st.title("AskBuddy - QnA Bot")
st.markdown("My QnA Bot with LangChain and Google Gemini !")  #description

if "messages" not in st.session_state:
	st.session_state.messages = []

for message in st.session_state.messages:
	role = message["role"]
	content = message["content"]
	st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything?")
if query:
	st.session_state.messages.append({"role":"user", "content":query})
	st.chat_message("user").markdown(query)

	res = llm.invoke(st.session_state.messages)
	answer = res.text if hasattr(res, "text") else res.content[0]["text"]

	st.chat_message("AI").markdown(answer)
	st.session_state.messages.append({"role":"ai", "content":answer})
