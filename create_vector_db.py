import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load all PDFs from data folder
loader = PyPDFDirectoryLoader("data")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Total Chunks: {len(docs)}")

for doc in docs:
    filename = os.path.basename(doc.metadata["source"]).lower()

    if filename == "cn.pdf":
        doc.metadata["subject"] = "Computer Networks"

    elif filename == "dbms.pdf":
        doc.metadata["subject"] = "Database Management System"

    elif filename == "ml.pdf":
        doc.metadata["subject"] = "Machine Learning"

    elif filename == "os.pdf":
        doc.metadata["subject"] = "Operating System"

    elif filename == "oop.pdf":
        doc.metadata["subject"] = "Object Oriented Programming"

    elif filename == "ds.pdf":
        doc.metadata["subject"] = "Data Structures"

    elif filename == "cyber.pdf":
        doc.metadata["subject"] = "Cyber Security"

# Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

import time
from langchain_community.vectorstores import FAISS

batch_size = 50
vector_db = None

for i in range(0, len(docs), batch_size):

    batch = docs[i:i + batch_size]

    print(f"Processing batch {i//batch_size + 1}...")

    if vector_db is None:
        vector_db = FAISS.from_documents(batch, embeddings)
    else:
        temp_db = FAISS.from_documents(batch, embeddings)
        vector_db.merge_from(temp_db)

    # Wait before the next batch
    time.sleep(45)

# Save FAISS
vector_db.save_local("faiss_index")

print("✅ FAISS Database Created Successfully!")