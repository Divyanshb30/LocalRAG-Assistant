# LocalRAG Assistant 

A powerful local AI knowledge assistant with RAG (Retrieval-Augmented Generation) capabilities and multi-tool support. Query your documents, analyze data, and automate tasks—all running locally with no API costs.


##  Features

- ** Document Q&A**: Ask questions about your documents using semantic search (FAISS vector database)
- ** Smart Summarization**: Automatically summarize long documents and extract key insights
- ** Task Breakdown**: Convert requirements into structured development tasks
- ** CSV Analysis**: Analyze data files with natural language queries
- ** 100% Local**: Runs entirely on your machine - no API keys, no internet required

---

## Tech Stack

- **LLM**: Qwen2.5-3B-Instruct (GGUF format via llama-cpp-python)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Framework**: Streamlit (web UI)
- **Tools**: pandas, pypdf, python-dateutil


