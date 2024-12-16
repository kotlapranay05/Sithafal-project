PDF Extraction and Interaction Using RAG Pipeline
Project Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that enables users to interact with semi-structured data from PDFs. It extracts, processes, stores the content in a vector database, and generates context-rich natural language responses based on user queries using a pre-trained Flan-T5 model.

Features
PDF Text Extraction: Extract text and tables from PDFs.
Text Chunking: Split extracted content into smaller chunks for better retrieval and query processing.
Sentence Embeddings: Use Sentence-Transformers to create vector embeddings from the extracted chunks.
FAISS Vector Database: Store and retrieve embeddings efficiently for similarity search.
Natural Language Response Generation: Generate accurate, context-driven responses using Flan-T5 based on the retrieved chunks of data.

Usage
Provide a PDF: The script will process the given PDF file to extract its text and structured data.
Ask a Query: Input your natural language query, and the system will search the vector database for relevant chunks. It will use Flan-T5 to generate an answer based on the retrieved information.


Example
Input:

PDF: unemployment_data.pdf (This is the PDF file to be processed)
Query: "What is the unemployment rate for Bachelor’s degree holders?"
Output:

"According to the data from the report, the unemployment rate for Bachelor’s degree holders is 4.5%."