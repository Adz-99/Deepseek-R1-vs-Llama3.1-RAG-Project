import os
import json
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

working_dir = os.getcwd()

# config_data = json.load(open(f"{working_dir}/src/config.json"))
# os.environ["GROQ_API_KEY"] = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY] = secrets.GROQ_API_KEY

# Setup llms and embedding
llm_deepseek = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0
)
llm_llama3 = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
embedding = HuggingFaceEmbeddings()

def process_document(file_name):
    # Load file
    full_path = os.path.join(working_dir, file_name)
    loader = UnstructuredPDFLoader(full_path)
    document = loader.load()

    # Split into text chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    text_chunks = splitter.split_documents(document)

    # Pass into vector db through embeddings
    vector_db = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore",
    )

    return 0

def get_answer(user_prompt):
    # Load vectordb
    vector_db = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )

    retriever = vector_db.as_retriever()

    # Get answer from each llm separately
    chain1 = RetrievalQA.from_chain_type(
        llm=llm_deepseek,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    chain2 = RetrievalQA.from_chain_type(
        llm=llm_llama3,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    deepseek_output = chain1.invoke({"query": user_prompt})
    deepseek_answer = deepseek_output["result"]
    llama_output = chain2.invoke({"query": user_prompt})
    llama_answer = llama_output["result"]

    return {"deepseek_answer": deepseek_answer, 
            "llama_answer": llama_answer}
