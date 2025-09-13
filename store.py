import pymupdf
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
vectorstore_name=os.getenv("VECTORSTORE_NAME")
text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
embedder = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
pdf_file_path=f"agreements\Tamilnadu Regulation of Rights and Responsibilities of Landlords and Tenants act.pdf"



# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = doc.pages()
    text = ""
    for page in pages:
        text += page.get_text()
    return text

# Chunk text
def chunk_text(text):
    return text_splitter.split_text(text)


# Ingest embeddings into FAISS
def ingest_to_faiss(chunks):
    faiss_vector_store = FAISS.from_texts(chunks, embedder)
    faiss_vector_store.save_local(vectorstore_name)
    print("Ingested in vector store ",faiss_vector_store)
    return faiss_vector_store


text = extract_text_from_pdf(pdf_file_path)
chunks = chunk_text(text)
faiss_vector_store = ingest_to_faiss(chunks)
print("Ingested in faiss vector store ",faiss_vector_store.index)
