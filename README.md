# AI Agents

A collection of small AI agent implementations built with LangChain and LangGraph. The repository starts with a basic LLM-powered Q&A bot and gradually adds tool usage, web search, SQL database interaction, RAG, memory, streaming, and LangGraph-based state management.

The applications are mainly exposed through Streamlit for experimentation and testing, while one of the LangGraph examples runs directly from the terminal.

## Repository Structure

```text
AI_agents/
├── 1_qna_bot.py
├── 2_google_agent.py
├── 3_qna_bot_with_groq.py
├── 4_sql_agent.py
├── 5_rag_agent.py
├── 6_langgraph_qna_bot.py
└── README.md
```

## Agents

### 1. Q&A Bot

`1_qna_bot.py`

A basic conversational Q&A application using LangChain and Google Gemini.

- Uses `ChatGoogleGenerativeAI`
- Uses the Gemini 3.7 Flash model
- Provides a Streamlit chat interface
- Maintains conversation history using Streamlit session state
- Sends the conversation history to the model for each response

This is the simplest implementation in the repository and serves as the starting point for the other agents.

### 2. Google Search Agent

`2_google_agent.py`

A LangChain agent that can use Google search as a tool.

- Uses `ChatGroq` with `openai/gpt-oss-20b`
- Uses `GoogleSerperAPIWrapper` for Google search
- Uses LangChain's `create_agent`
- Uses LangGraph's `MemorySaver` for checkpointing
- Runs from the terminal
- Maintains a conversation thread using a fixed thread ID

The agent decides when to use the search tool based on the user's query.

### 3. Q&A Bot with Groq and Web Search

`3_qna_bot_with_groq.py`

A Streamlit-based agent that combines an LLM, Google search, memory, and streamed responses.

- Uses `ChatGroq`
- Uses `openai/gpt-oss-20b`
- Uses `GoogleSerperAPIWrapper` as a search tool
- Uses LangChain's `create_agent`
- Uses LangGraph's `MemorySaver`
- Streams the model response into the Streamlit interface
- Maintains chat history through Streamlit session state

This builds on the previous search agent while moving the interaction into a web interface and adding response streaming.

### 4. SQL Agent

`4_sql_agent.py`

A task management agent that uses an LLM to interact with a SQLite database.

The application creates a `tasks` table with the following fields:

```text
id
title
description
status
created_at
```

The supported task statuses are:

```text
pending
in_progress
completed
```

The agent uses:

- `SQLDatabase`
- `SQLDatabaseToolkit`
- `ChatGroq`
- LangChain `create_agent`
- LangGraph `InMemorySaver`
- Streamlit

The agent can perform task-related CRUD operations through SQL tools. The system prompt also restricts SELECT results to a maximum of 10 records and instructs the agent to verify database changes after CREATE, UPDATE, and DELETE operations.

The database is stored locally as:

```text
my_tasks.db
```

### 5. RAG Agent

`5_rag_agent.py`

A PDF question-answering application using retrieval-augmented generation.

The application allows PDF files to be uploaded through Streamlit and processes them into a searchable in-memory vector store.

The document pipeline is:

```text
PDF files
    ↓
PyPDFDirectoryLoader
    ↓
Text splitting
    ↓
Google Gemini embeddings
    ↓
InMemoryVectorStore
    ↓
Similarity search
    ↓
Retrieved context
    ↓
Groq LLM
    ↓
Answer
```

The implementation uses:

- `PyPDFDirectoryLoader`
- `RecursiveCharacterTextSplitter`
- `GoogleGenerativeAIEmbeddings`
- `InMemoryVectorStore`
- `ChatGroq`
- LangChain tools and agents
- LangGraph `InMemorySaver`
- Streamlit

Documents are split into chunks of 2000 characters with a 200-character overlap. For a query, the retrieval tool performs similarity search and returns the four most relevant document chunks to the agent.

Uploaded documents are stored under:

```text
apps/doc_files/
```

### 6. LangGraph Q&A Bot

`6_langgraph_qna_bot.py`

A basic Q&A agent implemented directly using LangGraph's graph abstraction.

The graph contains:

```text
START
  ↓
chatbot
  ↓
END
```

The application defines a `ChatState` model containing the conversation messages and uses `add_messages` to manage message state.

It uses:

- `ChatGroq`
- `StateGraph`
- `START` and `END`
- `InMemorySaver`
- Pydantic `BaseModel`

The graph is compiled with a checkpointer so conversation state can be maintained using a thread ID.

Unlike the Streamlit applications, this example runs directly in the terminal.

## Technologies Used

- Python
- LangChain
- LangGraph
- Streamlit
- Groq
- Google Gemini
- Google Serper
- SQLite
- Pydantic

## Environment Variables

The scripts load environment variables from a `.env` file located relative to the project.

Depending on which agent is being used, API credentials are required for the corresponding services, including Groq, Google Gemini, and Google Serper.

Example:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
SERPER_API_KEY=your_serper_api_key
```

Use only the variables required by the agent you are running.

## Running the Streamlit Agents

Install the required Python packages for the agents you want to run, then start a Streamlit application with:

```bash
streamlit run 1_qna_bot.py
```

Other Streamlit agents can be started in the same way:

```bash
streamlit run 3_qna_bot_with_groq.py
streamlit run 4_sql_agent.py
streamlit run 5_rag_agent.py
```

## Running the Terminal Agents

The Google search agent can be started with:

```bash
python 2_google_agent.py
```

The LangGraph Q&A example can be started with:

```bash
python 6_langgraph_qna_bot.py
```

Both applications accept queries through the terminal and exit when:

```text
quit
```

is entered.

## Purpose

This repository is a collection of implementations for learning and experimenting with AI agents. Each file focuses on a different part of the agent stack, moving from direct LLM interaction to agents that can use external tools, interact with databases, retrieve information from documents, maintain state, and execute through LangGraph.
