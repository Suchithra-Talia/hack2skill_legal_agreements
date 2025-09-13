
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
from google import genai
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
vectorstore_name=os.getenv("VECTORSTORE_NAME")
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash-latest"
K_RETRIEVAL = 5
client = genai.Client(api_key=GEMINI_API_KEY)

def upload_agreement(file_path):

    uploaded_file = client.files.upload(file=file_path)
    # Poll or fetch final URI after processing if needed
    return uploaded_file


def load_faiss_index(index_path):
    embedder = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    faiss_index = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
    return faiss_index


def retrieve_context(faiss_index, regulation_prompt, k=K_RETRIEVAL):
    docs = faiss_index.similarity_search(regulation_prompt, k=k)
    #print(f"Retrieving context>>> {docs}")
    return [doc.page_content for doc in docs]

def validate_with_gemini(context_block, file_uri, user_prompt):

    system_instruction = (
        "You are a legal expert. "
        "You will have Relevant context about Rental Agreements from TamilNadu Regulation of Rights and Responsibilities of Landlords and Tenants."
        "with this knowledge, Read the file uploaded by the user."
        "Validate if uploaded document comply with data protection Rights and Responsibilities provided in the context."
        "Reference the full agreement if needed. Also clearly explain the user all the terms and conditions in layman terms so that the "
        "user is not having any ambiguity. "
    )
    prompt = (
        f"{system_instruction}\n\n"
        f"User prompt: {user_prompt}\n\n"
        f"Relevant extracted text:\n{context_block}\n"
    )

    # messages is a list of chat message dicts or HumanMessage types
    response = client.models.generate_content(
        model="gemini-1.5-flash-latest",
        contents=[prompt,file_uri]
    )
    #print(response)
    return response

    # response = llm.invoke([HumanMessage(content=prompt)])
    # return response.content

def main():
    user_prompt = input("\nEnter your legal validation question: ")
    agreement_file_path = f"agreements\chithra_Lease Agreement_Rental_Jan-1.pdf"
    file_uri = upload_agreement(agreement_file_path)
    faiss_index = None
    try:
        faiss_index = load_faiss_index(vectorstore_name)
        print("Faiss index loaded")
    except FileNotFoundError as e:
        print("Exception while loading the index..",e)
    if faiss_index:
        print("Faiss index loaded")
        context_chunks = retrieve_context(faiss_index, "Rights and Responsibilities of Landlords and Tenants")
        context_block = "\n\n".join(context_chunks)
    else:
        context_block = ""
    print("file uri >>> ",file_uri)
    result = validate_with_gemini(context_block, file_uri, user_prompt)
    print("\nGemini's validation response:\n")
    print(result.text)

if __name__ == "__main__":
    main()





    # 5. Validate laws using Gemini LLM
