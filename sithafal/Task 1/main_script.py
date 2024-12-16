!pip install PyPDF2 langchain sentence-transformers faiss-cpu pandas


# Import necessary modules
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from google.colab import files

# Step 1: Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Chunk the extracted text
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Step 3: Embed chunks using Sentence Transformers
def embed_text(chunks, model):
    return model.encode(chunks)

# Step 4: Add embeddings to FAISS vector store
def create_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Step 5: Search for similar chunks
def search_index(index, query_vector, k=5):
    distances, indices = index.search(query_vector, k)
    return indices, distances

# Step 6: Use a free LLM to generate responses
def generate_response(retrieved_chunks, query, qa_model):
    context = " ".join(retrieved_chunks)
    prompt = f"Based on the following information, answer the question: {query}\n\n{context}"
    response = qa_model(prompt)
    return response[0]['generated_text']

# Main script
if __name__ == "__main__":
    # Provide the PDF file path
    uploaded = files.upload()  # Manually upload the PDF
    pdf_path = list(uploaded.keys())[0]

    # Step 1: Extract text
    pdf_text = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk the text
    chunks = chunk_text(pdf_text)

    # Step 3: Generate embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_text(chunks, embedding_model)

    # Step 4: Create FAISS index
    index = create_faiss_index(embeddings)

    # Step 5: Handle user query
    user_query = input("Enter query :")
    query_vector = embedding_model.encode([user_query])
    indices, distances = search_index(index, query_vector)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    print("Relevant chunks retrieved:", retrieved_chunks)

    # Step 6: Generate response using a free LLM
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    response = generate_response(retrieved_chunks, user_query, qa_pipeline)
    print("Response:", response)
