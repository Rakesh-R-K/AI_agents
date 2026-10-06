from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path = env_path)

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import streamlit as st

if "document_uploaded" not in st.session_state:
	st.session_state.document_uploaded = False

if "agent" not in st.session_state:
	st.session_state.agent = None

if "vector_store" not in st.session_state:
	st.session_state.vector_store = None

if "messages" not in st.session_state:
	st.session_state.messages = []

def process_document(path:str):

	pdf_loader = PyPDFDirectoryLoader(path)
	docs = pdf_loader.load()

	splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
	splitted_text = splitter.split_documents(documents = docs)

	embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
	vector_db = InMemoryVectorStore.from_documents(
		documents = splitted_text,
		embedding = embeddings,
	)

	llm = ChatGroq(
		model = "openai/gpt-oss-20b"
	)

	@tool
	def retrieve_context(query:str):
		"""
			Retrieve documents relevant to a query from the knowledge base.
		"""
		context = ""
		docs = vector_db.similarity_search(query = query, k = 4)
		for i in docs:
			context += i.page_content + "\n\n"
		return context

	sys_prompt = """
			You are helpful assistant that answers questions using retrieved context.
			My knowledge base consists of the details from the uploaded document.
			ALWAYS use the `retrieve_context` tool for questions requiring external knowledge.
	"""

	memory = InMemorySaver()

	agent = create_agent(
		model = llm,
		tools = [retrieve_context],
		system_prompt = sys_prompt,
		checkpointer = memory,
	)
	st.session_state.agent = agent
	st.session_state.document_uploaded = True

st.title("PDFGPT - PDF AI CHATBOT")

if not st.session_state.document_uploaded:
	uploaded = st.file_uploader(label = "Select PDF File", type = ["pdf"], accept_multiple_files = True)
	if uploaded:
		with st.spinner("Processing..."):
			path = Path(__file__).resolve().parent.parent / 'apps' / 'doc_files'
			path.mkdir(parents = True, exist_ok = True)
			for file in uploaded:
				file_path = path / file.name
				with open(file_path, "wb") as f:
					f.write(file.getvalue())

			process_document(path)
			st.rerun()

if st.session_state.document_uploaded and st.session_state.agent:
	for message in st.session_state.messages:
		role = message.get("role")
		content = message.get("content")
		st.chat_message(role).markdown(content)

	query = st.chat_input("Ask Anything related to uploaded documents")
	if query:
		st.chat_message("user").markdown(query)
		st.session_state.messages.append({"role":"user", "content":query})
		response = st.session_state.agent.invoke(
				{"messages":[{"role":"user", "content":query}]},
				{"configurable":{"thread_id":"1"}}
		)
		answer = response["messages"][-1].content
		st.session_state.messages.append({"role":"ai", "content":answer})
		st.chat_message("ai").markdown(answer)



#while True:
#	query = input("User :")
#	if query.lower() == "quit":
#		print("Goodbye")
#		break
#
#	response = agent.invoke(
#			{"messages":[{"role":"user", "content":query}]},
#			{"configurable":{"thread_id":"RK1"}},
#		)
#	result = response["messages"][-1].content
#	print("AI :", result)
#
