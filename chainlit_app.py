import chainlit as cl
import os
# Import your business logic functions here
# (Assuming these are implemented as in your main())
from retriever import (
    upload_agreement,
    load_faiss_index,
    retrieve_context,
    validate_with_gemini

)
vectorstore_name=os.getenv("VECTORSTORE_NAME")

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Welcome! Please upload your agreement file to begin."
    ).send()

    files = await cl.AskFileMessage(
        content="Upload your agreement PDF.", accept=["application/pdf"]
    ).send()
    faiss_index = None
    file_uri=None
    # Defensive: check if files is not None and list not empty
    if files and len(files) > 0:
        file_path = files[0].path
        await cl.Message(f"File `{files[0].name}` uploaded!").send()
        # Proceed with your workflow...
        file_path = files[0].path
        print(f"file path is >> {file_path}")
        # Upload to Gemini and get URI
        file_uri = upload_agreement(file_path)
        print("File URI is >>> ", file_uri)

        # Try to load or build FAISS index
        await cl.Message(
            content="👋 File Uploaded.. Loading Index."
        ).send()
        try:
            faiss_index = load_faiss_index(vectorstore_name)
            print("Faiss index loaded")
        except FileNotFoundError as e:
            print("Exception while loading the index..", e)
        if faiss_index:
            print("Faiss index exists")
            context_chunks = retrieve_context(faiss_index, "Rights and Responsibilities of Landlords and Tenants")
            context_block = "\n\n".join(context_chunks)
            print("context_block >>> ", context_block)
        else:
            context_block = ""
        print("file uri >>> ", file_uri)
        cl.user_session.set("file_uri", file_uri)
        cl.user_session.set("faiss_index", faiss_index)
        cl.user_session.set("context_block", context_block)
        await cl.Message("✅ All set!! Ask your legal question.").send()
    else:
        await cl.Message("File upload not detected. Please upload a PDF!").send()


@cl.on_message
async def chat(message: cl.Message):
    user_prompt = message.content
    print("in chat...")
    file_uri = cl.user_session.get("file_uri")
    faiss_index = cl.user_session.get("faiss_index")
    if not (faiss_index and file_uri):
        await cl.Message("Session invalid. Please start a new chat and upload an agreement.").send()
        return
    print("am here..")
    # Retrieve relevant context for the query
    # context_chunks = retrieve_context(faiss_index, user_prompt)
    # context_block = "\n\n".join(context_chunks)
    context_block = cl.user_session.get("context_block")
    # Validate using Gemini LLM
    result = validate_with_gemini(context_block, file_uri, user_prompt)
    await cl.Message(f"**Legal Bot Response:**\n\n{result.text}").send()
