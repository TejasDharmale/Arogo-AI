# Patient Summary API

This FastAPI-based project processes patient data and generates layperson-friendly summaries based on provided health conditions. It leverages advanced natural language processing (NLP) using Hugging Face models and similarity search via FAISS.

## Features
- **Patient Data Retrieval**: Retrieves the most relevant patient data based on the provided query.
- **Text Summarization**: Generates a summary of patient health information in easy-to-understand language.
- **Interactive API Documentation**: Explore and test the API via Swagger UI (`/docs`).

## Technology Stack
- **FastAPI**: Framework for building APIs.
- **Hugging Face Transformers**: Used for text generation (`distilgpt2` model).
- **Sentence Transformers**: Generates embeddings for similarity search.
- **FAISS**: Efficient similarity search for patient data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/patient-summary-api.git
   cd patient-summary-api
