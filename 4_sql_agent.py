from pathlib import Path
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path = env_path)

db = SQLDatabase.from_uri("sqlite:///my_tasks.db")
db.run("""
	CREATE TABLE IF NOT EXISTS tasks (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	title TEXT NOT NULL,
	description TEXT,
	status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	);
""")

llm = ChatGroq(model = "openai/gpt-oss-20b")

toolkit = SQLDatabaseToolkit(db = db, llm = llm)
tools = toolkit.get_tools()
if "memory" not in st.session_state:
	st.session_state.memory = InMemorySaver()
	st.session_state.history = []

#for tool in tools:
	#print(tool.name)
#sql_db_query
#sql_db_schema
#sql_db_list_tables
#sql_db_query_checker

sys_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
	CREATE: INSERT INTO tasks(title, description, status)
	READ: SELECT * FROM tasks WHERE ...LIMIT 10
	UPDATE: UPDATE tasks SET status=? where id=? OR title=?
	DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""

@st.cache_resource  # This will not call everytime streamlit runs the code again and again.
def get_agent():
	agent = create_agent(
		model = llm,
		tools = tools,
		checkpointer = st.session_state.memory,
		system_prompt = sys_prompt,
	)
	return agent

agent = get_agent()

st.title("TaskBot - Manage your todos")

for message in st.session_state.history:
	role = message["role"]
	content = message["content"]
	st.chat_message(role).markdown(content)

prompt = st.chat_input("Ask me to manage your tasks?")
if prompt:
	st.chat_message("user").markdown(prompt)
	st.session_state.history.append({"role":"user", "content":prompt})
	with st.chat_message("ai"):
		with st.spinner("Processing..."):
			response = agent.invoke(
				{"messages":[{"role":"user", "content":prompt}]},
				{"configurable":{"thread_id":"RK1"}}
			)
			result = response["messages"][-1].content
			st.session_state.history.append({"role":"ai", "content":result})
			st.markdown(result)
