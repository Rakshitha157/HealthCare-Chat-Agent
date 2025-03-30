import os
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PyPDF2 import PdfReader
import docx2txt

# Source of the data
PDF_DIRECTORY = r'/Users/rakshithajaganth/Downloads/ecommerce.py'
os.makedirs(PDF_DIRECTORY, exist_ok=True)

# Loading the data from the source
def load_documents(directory):
    documents = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith(".pdf"):
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                documents.append(text)
        elif file.endswith(".docx") or file.endswith(".doc"):
            text = docx2txt.process(file_path)
            documents.append(text)
        elif file.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    return documents

# Data embedding, chunkking, and loading into database
def create_faiss_index(documents):
    if not documents:
        return None, None, None
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    document_embeddings = np.array([embedding_model.encode(doc) for doc in documents])
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)
    faiss.write_index(index, 'faiss_index.index')
    np.save('document_chunks.npy', documents)
    return index, documents, embedding_model

# Loading the database
def load_existing_index():
    if os.path.exists('faiss_index.index') and os.path.exists('document_chunks.npy'):
        index = faiss.read_index('faiss_index.index')
        documents = np.load('document_chunks.npy', allow_pickle=True)
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return index, documents, embedding_model
    return None, None, None

# Qurey and Respones 
def query_knowledge_base(query, index, embedding_model, document_texts, top_n=3):
    if index is None:
        return "No knowledge base found. Please upload documents first.", []
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_n)
    results = [(document_texts[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    top_documents = " ".join([doc for doc, _ in results])
    result = pipe(f"Context: {top_documents}\nQuestion: {query}")
    return result[0]['generated_text'], results

# Defining the indexes
index, document_texts, embedding_model = load_existing_index()
if index is None:
    document_texts = load_documents(PDF_DIRECTORY)
    index, document_texts, embedding_model = create_faiss_index(document_texts)

# Pretrained tecxt to text model for th agent 
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Interface for the agnet
def main():
    st.title("Health Care Agent")
    query = st.text_input("How can I assist you today?")
    if query == "hi":
        st.subheader("AI Response:")
        st.write("Hello! How can i assist you today?")
    elif query == "how are you":
        st.subheader("AI Response:")
        st.write("I am good, How do you do, What can I do for you?")
    if query:
        answer, relevant_docs = query_knowledge_base(query, index, embedding_model, document_texts)
        st.subheader("AI Response:")
        st.write(answer)
        
        if relevant_docs:
            st.subheader("Relevant Documents:")
            for doc, _ in relevant_docs:
                st.write("- ", doc[:200] + "...")  # Show only preview

if __name__ == "__main__":
    main()
