import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Initialize embedding model and vector index
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained embedding model
dimension = 384  # Embedding dimension for MiniLM
index = faiss.IndexFlatL2(dimension)  # FAISS flat index
metadata_store = []  # Store chunks and metadata

def crawl_and_scrape(url):
    """Crawl and scrape content from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])  # Extract paragraph texts
    return text

def chunk_text(text, chunk_size=500):
    """Divide text into manageable chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_and_store(url):
    """Crawl, scrape, chunk, and store data in the vector database."""
    global metadata_store
    text = crawl_and_scrape(url)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    faiss.normalize_L2(embeddings)
    index.add(np.array(embeddings))
    metadata_store.extend([{'text': chunk} for chunk in chunks])
    print(f"Processed {len(chunks)} chunks from {url}")

def query_database(query, top_k=3):
    """Search the vector database for relevant chunks."""
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = [metadata_store[idx]['text'] for idx in indices[0] if idx < len(metadata_store)]
    return results

# Initialize the language model pipeline
llm = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_response(query, retrieved_texts):
    """Generate a response using the language model."""
    context = "\n".join(retrieved_texts)
    prompt = f"Based on the following information, answer the query:\n\n{context}\n\nQuery: {query}\n\nAnswer:"
    response = llm(prompt, max_length=300, num_return_sequences=1)
    return response[0]['generated_text']

# Main workflow
if __name__ == "__main__":
    url = input("Enter the URL to scrape: ")
    process_and_store(url)

    query = input("Enter your query: ")
    relevant_chunks = query_database(query)
    if relevant_chunks:
        response = generate_response(query, relevant_chunks)
        print("Response:", response)
    else:
        print("No relevant information found.")
