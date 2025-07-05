import pandas as pd
import os
import torch
from tqdm import tqdm
import langchain; 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class ComplaintChunkEmbedder:
    def __init__(
        self,
        data_path: str,
        persist_directory: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        sample_size: int = None
    ):
        self.data_path = data_path
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sample_size = sample_size
        self.df = None
        self.documents = None
        self.embedding_model = None
        self.vector_store = None

    def load_data(self):
        print(f"📥 Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        if self.sample_size:
            df = df.sample(self.sample_size, random_state=42).reset_index(drop=True)
            print(f"🔍 Sampled {self.sample_size} records.")
        else:
            print(f"✅ Loaded full dataset with {len(df)} records.")
        self.df = df

    def chunk_documents(self):
        print("✂️  Splitting narratives into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        docs = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Chunking"):
            complaint_id = row.get("complaint_id", str(i))
            product = row.get("product", "Unknown")
            text = str(row.get("cleaned_narrative", "")).strip()

            if not text:
                continue

            chunks = splitter.split_text(text)
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata={"complaint_id": complaint_id, "product": product}
                ))

        print(f"📄 Generated {len(docs)} document chunks.")
        self.documents = docs

    def load_embedding_model(self):
        print("⚙️ Loading embedding model: intfloat/e5-small-v2")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/e5-small-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )

    def create_and_save_vectorstore(self):
        print("🧠 Creating vector store with Chroma...")
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()
        print(f"✅ Vector store saved to {self.persist_directory}")

  


