import streamlit as st
from llama_cpp import Llama
from rag import RAGPipeline
from tools import get_tools
from agent import RAGAgent
import os
import time
import pickle

# Page config
st.set_page_config(
    page_title="Local AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'rag' not in st.session_state:
    st.session_state.rag = None

def process_uploaded_file(uploaded_file, file_path):
    """Process uploaded files based on type"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        import pypdf
        pdf_reader = pypdf.PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        
        # Save as txt for RAG processing
        txt_path = file_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return txt_path
    
    return file_path

@st.cache_resource
def load_system():
    """Load RAG, LLM, and Agent (cached)"""
    with st.spinner("🔄 Loading AI system... (this takes ~30 seconds)"):
        # Load RAG
        rag = RAGPipeline()
        if os.path.exists("vector_store/index.faiss"):
            rag.load_index("vector_store")
        else:
            st.warning("⚠️ No vector store found. Please upload documents first.")
        
        # Load LLM
        llm = Llama(
            model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
            n_ctx=2048,
            verbose=False
        )
        
        # Initialize tools and agent
        tools = get_tools(rag)
        agent = RAGAgent(llm, tools)
        
        return agent, rag

# Sidebar
with st.sidebar:
    st.title("🤖 AI Assistant")
    st.markdown("---")
    
    # Document Upload
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload TXT, PDF, or CSV files",
        type=['txt', 'pdf', 'csv'],
        help="Upload documents to add to knowledge base"
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]
        
        # Save uploaded file
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Process PDF if needed
        if file_type.lower() == 'pdf':
            with st.spinner("Processing PDF..."):
                process_uploaded_file(uploaded_file, file_path)
                st.success("✅ PDF processed!")
        
        # Rebuild index
        if st.button("🔄 Rebuild Knowledge Base"):
            with st.spinner("Rebuilding vector index..."):
                rag = RAGPipeline()
                rag.load_documents("data")
                rag.build_index()
                rag.save_index()
                st.success("✅ Knowledge base updated!")
                st.cache_resource.clear()
                st.rerun()
    
        # In the sidebar, after file upload
    if uploaded_file and file_type.lower() == 'csv':
        st.info("💡 CSV file detected! You can now ask data analysis questions.")
        st.session_state.csv_path = file_path

    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 System Info")
    if os.path.exists("vector_store/index.faiss"):
        try:
            with open("vector_store/chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
            st.metric("Document Chunks", len(chunks))
        except:
            st.metric("Document Chunks", "N/A")
    else:
        st.info("No knowledge base found")
    
    # Clear chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("💡 Example Queries")
    
    example_queries = [
        "What products does TechCorp offer?",
        "Summarize the company overview",
        "Create tasks to implement a new API",
        "What are the operating hours?"
    ]
    
    for query in example_queries:
        if st.button(query, key=query):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

# Main content
st.title("💬 Local AI Knowledge Assistant")
st.markdown("Ask questions about your documents using RAG + AI Agent")

# Load system
if st.session_state.agent is None:
    try:
        agent, rag = load_system()
        st.session_state.agent = agent
        st.session_state.rag = rag
        st.success("✅ AI System Ready!")
    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        st.info("Make sure you have run `python rag.py` first to build the vector store.")
        st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "tool" in message:
            st.caption(f"🔧 Tool used: **{message['tool']}**")
        if "response_time" in message:
            st.caption(f"⏱️ Response time: **{message['response_time']:.2f}s**")

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            start_time = time.time()
            
            csv_path = st.session_state.get('csv_path', None)

            result = st.session_state.agent.execute(prompt, csv_path=csv_path)
            
            response_time = time.time() - start_time
        
        # Stream the answer word-by-word (ChatGPT effect)
        full_answer = result['answer']
        words = full_answer.split()
        displayed_text = ""
        
        import time as t
        for word in words:
            displayed_text += word + " "
            message_placeholder.markdown(displayed_text + "▌")
            t.sleep(0.04)  # Streaming speed
        
        # Show final answer without cursor
        message_placeholder.markdown(full_answer)
        
        # Display metadata
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"🔧 Tool: **{result['tool_used']}**")
        with col2:
            st.caption(f"⏱️ Response time: {response_time:.2f}s")
        
        # Show raw output in expander
        with st.expander("🔍 View detailed output"):
            st.json(result['raw_output'])
    
    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
        "tool": result['tool_used'],
        "response_time": response_time
    })

# Footer
st.markdown("---")
st.caption("🚀 Powered by: Qwen2.5-3B | FAISS | Sentence Transformers | Local CPU")
