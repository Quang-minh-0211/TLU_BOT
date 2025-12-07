# app.py - TLU Chatbot với giao diện ChatGPT-like
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core. output_parsers import StrOutputParser
import time
from datetime import datetime
import uuid

# ============== CẤU HÌNH ==============
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PROMPT_TEMPLATE = """
Bạn là TLUBot - trợ lý tư vấn tuyển sinh thông minh của Trường Đại học Thủy lợi (Thuy Loi University - TLU). 

NGUYÊN TẮC TRẢ LỜI:
1.  Trả lời dựa trên thông tin trong phần "Thông tin tham khảo" bên dưới
2. Nếu câu hỏi về quy trình/các bước, hãy liệt kê ĐẦY ĐỦ tất cả các bước
3.  Trả lời thân thiện, chuyên nghiệp, dễ hiểu
4. Nếu không có thông tin, nói "Tôi không có thông tin về vấn đề này trong dữ liệu hiện tại.  Bạn có thể liên hệ trực tiếp với phòng Tuyển sinh của trường để được hỗ trợ."
5.  KHÔNG bịa thông tin

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""

# ============== KHỞI TẠO ==============
@st.cache_resource
def load_vectorstore():
    """Load ChromaDB - cached để không load lại mỗi lần"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

@st.cache_resource
def create_rag_chain(_vectorstore):
    """Tạo RAG chain - cached"""
    llm = OllamaLLM(
        model="qwen2.5:7b",
        temperature=0.1,
        num_predict=2048,
    )
    
    retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ============== CUSTOM CSS ==============
def load_css():
    st. markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background-color: #343541;
    }
    
    . stApp {
        background-color: #343541;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #202123;
        border-right: 1px solid #4d4d4f;
    }
    
    [data-testid="stSidebar"] . stMarkdown {
        color: #ececf1;
    }
    
    /* Chat container */
    . chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Message styles */
    .user-message {
        background-color: #343541;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        display: flex;
        gap: 15px;
    }
    
    . assistant-message {
        background-color: #444654;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        display: flex;
        gap: 15px;
    }
    
    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    
    . user-avatar {
        background-color: #5436DA;
        color: white;
    }
    
    .bot-avatar {
        background-color: #19C37D;
        color: white;
    }
    
    . message-content {
        color: #ececf1;
        line-height: 1.7;
        flex-grow: 1;
    }
    
    /* Input styling */
    . stTextInput > div > div > input {
        background-color: #40414f;
        border: 1px solid #565869;
        border-radius: 12px;
        color: #ececf1;
        padding: 15px 20px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #19C37D;
        box-shadow: 0 0 0 1px #19C37D;
    }
    
    . stTextInput > div > div > input::placeholder {
        color: #8e8ea0;
    }
    
    /* Button styling */
    . stButton > button {
        background-color: #19C37D;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #1a9d6c;
        transform: translateY(-1px);
    }
    
    /* New chat button */
    . new-chat-btn {
        background-color: transparent;
        border: 1px solid #565869;
        color: #ececf1;
        padding: 12px 15px;
        border-radius: 8px;
        width: 100%;
        text-align: left;
        cursor: pointer;
        transition: all 0. 2s;
        margin-bottom: 10px;
    }
    
    . new-chat-btn:hover {
        background-color: #2a2b32;
    }
    
    /* Chat history item */
    .chat-history-item {
        padding: 10px 15px;
        border-radius: 8px;
        color: #ececf1;
        cursor: pointer;
        transition: all 0. 2s;
        margin: 2px 0;
        font-size: 14px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .chat-history-item:hover {
        background-color: #2a2b32;
    }
    
    /* Welcome screen */
    .welcome-container {
        text-align: center;
        padding: 60px 20px;
        color: #ececf1;
    }
    
    . welcome-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #ececf1;
    }
    
    .welcome-subtitle {
        font-size: 18px;
        color: #8e8ea0;
        margin-bottom: 40px;
    }
    
    /* Example cards */
    .example-cards {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .example-card {
        background-color: #40414f;
        border: 1px solid #565869;
        border-radius: 12px;
        padding: 15px;
        cursor: pointer;
        transition: all 0.2s;
        text-align: left;
    }
    
    .example-card:hover {
        background-color: #4a4b59;
        border-color: #19C37D;
    }
    
    .example-card-title {
        color: #ececf1;
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    . example-card-desc {
        color: #8e8ea0;
        font-size: 12px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2a2b32;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #565869;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6e6e80;
    }
    
    /* Loading animation */
    .typing-indicator {
        display: flex;
        gap: 5px;
        padding: 10px 0;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: #8e8ea0;
        border-radius: 50%;
        animation: typing 1. 4s infinite;
    }
    
    . typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.4;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .example-cards {
            grid-template-columns: 1fr;
        }
        
        .welcome-title {
            font-size: 24px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============== COMPONENTS ==============
def render_message(role, content, avatar):
    """Render một message với style ChatGPT"""
    if role == "user":
        message_class = "user-message"
        avatar_class = "user-avatar"
    else:
        message_class = "assistant-message"
        avatar_class = "bot-avatar"
    
    st.markdown(f"""
    <div class="{message_class}">
        <div class="message-avatar {avatar_class}">{avatar}</div>
        <div class="message-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def render_welcome():
    """Render màn hình welcome"""
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">🎓 TLU Chatbot</div>
        <div class="welcome-subtitle">Trợ lý tư vấn tuyển sinh Trường Đại học Thủy lợi</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Example questions
    col1, col2 = st.columns(2)
    
    examples = [
        ("📊 Điểm chuẩn", "Điểm chuẩn ngành CNTT năm 2024? "),
        ("💰 Học phí", "Học phí các ngành là bao nhiêu?"),
        ("📝 Xét tuyển", "Có những phương thức xét tuyển nào?"),
        ("🎁 Học bổng", "Điều kiện nhận học bổng là gì?"),
    ]
    
    with col1:
        for title, question in examples[:2]:
            if st.button(f"{title}\n{question}", key=question, use_container_width=True):
                return question
    
    with col2:
        for title, question in examples[2:]:
            if st. button(f"{title}\n{question}", key=question, use_container_width=True):
                return question
    
    return None

def render_sidebar():
    """Render sidebar với lịch sử chat"""
    with st.sidebar:
        # Logo và tiêu đề
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 40px;">🎓</div>
            <div style="color: #ececf1; font-size: 18px; font-weight: 600; margin-top: 10px;">TLU Chatbot</div>
            <div style="color: #8e8ea0; font-size: 12px;">Đại học Thủy lợi</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Nút tạo chat mới
        if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
            st.session_state.messages = []
            st. session_state.current_chat_id = str(uuid.uuid4())
            st. rerun()
        
        st.markdown("---")
        
        # Lịch sử chat
        st.markdown("""
        <div style="color: #8e8ea0; font-size: 12px; padding: 10px 0;">
            📜 LỊCH SỬ TRÒ CHUYỆN
        </div>
        """, unsafe_allow_html=True)
        
        if "chat_history" in st.session_state and st.session_state.chat_history:
            for chat in st.session_state.chat_history[-10:]:  # Hiển thị 10 chat gần nhất
                chat_title = chat.get("title", "Cuộc trò chuyện")[:30]
                if st.button(f"💬 {chat_title}", key=chat["id"], use_container_width=True):
                    st.session_state. current_chat_id = chat["id"]
                    st.session_state.messages = chat. get("messages", [])
                    st.rerun()
        else:
            st. markdown("""
            <div style="color: #565869; font-size: 13px; padding: 10px;">
                Chưa có lịch sử trò chuyện
            </div>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="color: #565869; font-size: 11px; text-align: center; padding: 10px;">
            Powered by LangChain + Ollama<br>
            © 2024 TLU Chatbot
        </div>
        """, unsafe_allow_html=True)

def stream_response(response_text):
    """Giả lập streaming response"""
    for char in response_text:
        yield char
        time. sleep(0.01)

# ============== MAIN APP ==============
def main():
    # Page config
    st.set_page_config(
        page_title="TLU Chatbot - Tư vấn tuyển sinh",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS
    load_css()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st. session_state.chat_history = []
    
    if "current_chat_id" not in st. session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())
    
    # Load RAG components
    vectorstore = load_vectorstore()
    rag_chain = create_rag_chain(vectorstore)
    
    # Render sidebar
    render_sidebar()
    
    # Main chat area
    chat_container = st. container()
    
    with chat_container:
        # Nếu chưa có message, hiển thị welcome screen
        if not st.session_state. messages:
            selected_example = render_welcome()
            if selected_example:
                st.session_state.messages. append({
                    "role": "user",
                    "content": selected_example
                })
                st.rerun()
        else:
            # Hiển thị lịch sử chat
            for message in st.session_state.messages:
                if message["role"] == "user":
                    render_message("user", message["content"], "👤")
                else:
                    render_message("assistant", message["content"], "🤖")
    
    # Input area
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)  # Spacer
    
    # Chat input
    if prompt := st.chat_input("Nhập câu hỏi về tuyển sinh Đại học Thủy lợi... "):
        # Add user message
        st.session_state. messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        render_message("user", prompt, "👤")
        
        # Generate response
        with st.spinner(""):
            # Show typing indicator
            typing_placeholder = st.empty()
            typing_placeholder.markdown("""
            <div class="assistant-message">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get response from RAG
            response = rag_chain. invoke(prompt)
            
            # Clear typing indicator
            typing_placeholder.empty()
        
        # Add assistant message
        st.session_state.messages. append({
            "role": "assistant",
            "content": response
        })
        
        # Display assistant message with streaming effect
        message_placeholder = st.empty()
        full_response = ""
        for char in stream_response(response):
            full_response += char
            message_placeholder. markdown(f"""
            <div class="assistant-message">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="message-content">{full_response}▌</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Final message without cursor
        message_placeholder.markdown(f"""
        <div class="assistant-message">
            <div class="message-avatar bot-avatar">🤖</div>
            <div class="message-content">{response}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Save to chat history
        if st.session_state.messages:
            chat_title = st.session_state.messages[0]["content"][:50]
            
            # Update or add to history
            chat_exists = False
            for chat in st.session_state.chat_history:
                if chat["id"] == st.session_state.current_chat_id:
                    chat["messages"] = st.session_state.messages
                    chat["updated_at"] = datetime.now().isoformat()
                    chat_exists = True
                    break
            
            if not chat_exists:
                st.session_state.chat_history.append({
                    "id": st.session_state. current_chat_id,
                    "title": chat_title,
                    "messages": st.session_state.messages,
                    "created_at": datetime. now().isoformat(),
                    "updated_at": datetime. now().isoformat()
                })

if __name__ == "__main__":
    main()