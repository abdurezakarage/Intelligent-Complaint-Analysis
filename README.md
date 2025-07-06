# Intelligent-Complaint-Analysis
A comprehensive system for analyzing and processing customer complaints using advanced NLP techniques, including data preprocessing, embedding generation, and Retrieval-Augmented Generation (RAG) for intelligent complaint analysis.

## 🎯 Project Overview

This project implements an end-to-end pipeline for intelligent complaint analysis that can:
- Preprocess and clean large-scale complaint datasets
- Generate embeddings for semantic search
- Provide intelligent answers to questions about complaints using RAG
- Support multiple financial products (credit cards, loans, BNPL, etc.)

## 🏗️ Architecture

The project consists of three main components:

### 1. Data Preprocessing (`src/preprocess.py`)
- Loads and explores complaint data
- Filters for relevant financial products
- Cleans and normalizes complaint narratives
- Generates visualizations for data insights

### 2. Embedding & Indexing (`src/embed_index.py`)
- Splits complaint narratives into chunks
- Generates embeddings using HuggingFace models
- Creates a vector store using ChromaDB for semantic search

### 3. RAG Pipeline (`src/rag_pipline.py`)
- Implements Retrieval-Augmented Generation
- Retrieves relevant complaint chunks based on queries
- Generates intelligent answers using language models

## 📁 Project Structure

```
Intelligent-Complaint-Analysis/
├── data/                          # Data files
│   ├── complaints.csv             # Raw complaint data (5.6GB)
│   └── filtered_complaints.csv    # Preprocessed data (190MB)
├── src/                           # Source code
│   ├── preprocess.py              # Data preprocessing module
│   ├── embed_index.py             # Embedding and indexing module
│   ├── rag_pipline.py             # RAG pipeline implementation
│   └── __init__.py
├── notbooks/                      # Jupyter notebooks
│   ├── preprocess.ipynb           # Data preprocessing notebook
│   ├── embeding.ipynb             # Embedding generation notebook
│   └── rag.ipynb                  # RAG implementation notebook
├── vectorstore/                   # Vector database storage
│   └── chroma_db/                 # ChromaDB files
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```


### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Intelligent-Complaint-Analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Supported Financial Products

The system is configured to analyze complaints for the following products:
- Credit card
- Personal loan
- Buy Now, Pay Later (BNPL)
- Savings account
- Money transfer, virtual currency, or mobile wallet

## 🔧 Configuration

### Model Settings

- **Embedding Model**: `intfloat/e5-small-v2` (default)
- **Language Model**: `google/flan-t5-base` (default)
- **Chunk Size**: 300 tokens (configurable)
- **Chunk Overlap**: 50 tokens (configurable)
- **Top-k Retrieval**: 5 chunks (configurable)

### Performance Optimization

- Enable CUDA for GPU acceleration (automatic detection)
- Adjust `sample_size` for testing with smaller datasets
- Modify chunk parameters based on your data characteristics

## 📈 Data Processing Pipeline

1. **Raw Data**: Load complaints from CSV (5.6GB dataset)
2. **Filtering**: Select relevant financial products
3. **Cleaning**: Remove special characters, normalize text
4. **Chunking**: Split narratives into manageable pieces
5. **Embedding**: Generate semantic embeddings
6. **Indexing**: Store in ChromaDB vector database
7. **Querying**: Use RAG for intelligent analysis

## 🎓 Jupyter Notebooks

The project includes comprehensive Jupyter notebooks for interactive exploration:

- **`preprocess.ipynb`**: Step-by-step data preprocessing
- **`embeding.ipynb`**: Embedding generation and analysis
- **`rag.ipynb`**: RAG pipeline implementation and testing
