Chat with Website Using RAG Pipeline

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline for querying and interacting with website data. The system crawls web content, processes it into embeddings, stores it in a vector database, and uses a language model to generate context-rich responses.

---

Features
- Crawl and scrape website content.
- Store and retrieve data using FAISS vector database.
- Use a pre-trained SentenceTransformer for embeddings.
- Generate natural language responses using Flan-T5.

---

Setup Instructions

1. Clone the Repository
    git clone https://github.com/your-username/RAG-Pipeline-Project.git
    cd RAG-Pipeline-Project

2.Install Dependencies
    pip install -r requirements.txt

3.Run the Script
    python main_script.py


Usage
    Provide a URL: The script will scrape and process the content from the given website.
    Ask a Query: Input your natural language question, and the system will respond using the retrieved and processed web content.


Example
Input:
    URL: https://example.com/article
    Query: What are the key takeaways of the article?
Output:
    "The article discusses the importance of data privacy, emerging trends in AI, and actionable steps for businesses."