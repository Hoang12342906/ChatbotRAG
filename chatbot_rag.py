import os
import time
import logging
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# Load biến môi trường
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Giới hạn tìm kiếm Google
search_count = 0
MAX_SEARCHES_PER_DAY = 5  # Giới hạn tìm kiếm mỗi ngày
RESET_INTERVAL = 86400  # 24 giờ (tính bằng giây)
last_reset_time = time.time()

def can_search():
    """ Kiểm tra xem chatbot có thể tìm kiếm trên Google không. """
    global search_count, last_reset_time
    if time.time() - last_reset_time > RESET_INTERVAL:
        search_count = 0
        last_reset_time = time.time()
    return search_count < MAX_SEARCHES_PER_DAY

def increment_search_count():
    """ Tăng số lần tìm kiếm nếu chưa đạt giới hạn. """
    global search_count
    if can_search():
        search_count += 1
        return True
    return False

# Tải dữ liệu văn bản
def load_documents(directory="data"):
    os.makedirs(directory, exist_ok=True)
    if not os.listdir(directory):
        with open(f"{directory}/sample.txt", "w", encoding="utf-8") as f:
            f.write("Đây là một số thông tin mẫu về RAG.")
    try:
        loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=lambda file_path: TextLoader(file_path, encoding="utf-8"))
        return loader.load()
    except Exception as e:
        logger.error(f"Lỗi khi tải tài liệu: {e}")
        return []

# Chia nhỏ tài liệu
def split_documents(documents):
    if not documents:
        return []
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)

# Tạo vector store
def create_vector_store(chunks, persist_directory="vector_store"):
    if not chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)

# Khởi tạo Hugging Face AI
def initialize_llm():
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2", 
        token=os.getenv("HUGGINGFACE_API_KEY")
    )
    
    def generate_response(input_value):
        # Kiểm tra và xử lý đầu vào
        if isinstance(input_value, dict):
            context = input_value.get('context', '')
            question = input_value.get('question', '')
        elif isinstance(input_value, str):
            context = ''
            question = input_value
        else:
            context = ''
            question = str(input_value)
        
        # Tạo prompt theo template ban đầu
        full_prompt = f"""
        Bạn là một trợ lý AI chính xác và chuyên nghiệp. Hãy trả lời **thẳng vào câu hỏi** dựa trên dữ liệu có sẵn.  
        Nếu không tìm thấy thông tin, hãy **nói rõ rằng bạn không biết** thay vì đoán bừa.  

        📌 **Quy tắc trả lời**:  
        1️⃣ **Trả lời trực tiếp**, không lan man.  
        2️⃣ **Không thêm thông tin ngoài lề**.  
        3️⃣ **Nếu có thể, trích dẫn nguồn dữ liệu**.  
        4️⃣ **Nếu không biết, hãy nói thẳng rằng không có thông tin.**  

        🔎 **Dữ liệu hỗ trợ**:  
        {context}  

        📢 **Câu hỏi**: {question}  
        🎯 **Trả lời chính xác**:  
        """
        
        # Sử dụng InferenceClient để sinh văn bản
        response = client.text_generation(
            full_prompt, 
            max_new_tokens=150, 
            temperature=0.1
        )
        return response

    return RunnableLambda(generate_response)

# Thiết lập Google Search
def setup_google_search():
    return Tool(
        name="Google Search",
        func=SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_KEY")).run,
        description="Tìm kiếm thông tin trên Google."
    )

# Thiết lập chatbot
def setup_rag():
    documents = load_documents()
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)

    if not vector_store:
        raise ValueError("Không thể khởi tạo vector store.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = initialize_llm()
    google_search = setup_google_search()

    # Hàm truy xuất dữ liệu trước khi tìm kiếm trên Google
    def retrieve_and_search(query):
        # Nếu người dùng chỉ chào hỏi, trả về phản hồi đơn giản mà không truy xuất dữ liệu
        greetings = ["xin chào", "chào bạn", "hello", "hi", "yo", "hey","chào"]
        if query.lower() in greetings:
            return "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        if not context.strip():  
            if can_search():
                print("🔍 Không tìm thấy trong dữ liệu nội bộ. Đang tìm kiếm trên Google...")
                increment_search_count()
                return google_search.run(query)
            else:
                return "⚠️ Đã đạt giới hạn tìm kiếm trên Google hôm nay."
        
        return context

    return (
        {"context": RunnablePassthrough() | retrieve_and_search, "question": RunnablePassthrough()}
        | llm
    )

# Chạy chatbot
def main():
    print("🤖 Đang khởi tạo chatbot RAG với Hugging Face API...")
    try:
        rag_chain = setup_rag()
        print("\nChatbot đã sẵn sàng! Gõ 'exit' để thoát.")

        while True:
            user_input = input("\nBạn: ")

            if user_input.lower() == "exit":
                print("Tạm biệt! 👋")
                break

            print("\nĐang xử lý câu hỏi của bạn...")
            response = rag_chain.invoke(user_input)
            print(f"\nChatbot: {response}")

    except Exception as e:
        logger.error(f"Lỗi: {e}")
        print(f"Không thể khởi tạo chatbot. Lỗi: {e}")

if __name__ == "__main__":
    main()