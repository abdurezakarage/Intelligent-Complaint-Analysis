from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch

class RAGPipeline:
    def __init__(
        self,
        vectorstore_path: str,
        embedding_model_name: str = "intfloat/e5-small-v2",
        llm_model_name: str = "google/flan-t5-base",
        top_k: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.top_k = top_k
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device}
        )
        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embedding_model
        )
        self.generator = pipeline("text-generation", model=llm_model_name, device=0 if device == "cuda" else -1)

    def retrieve(self, query: str) -> List[str]:
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        return [doc.page_content for doc in docs]

    def build_prompt(self, context_chunks: List[str], query: str) -> str:
        context = "\n\n".join(context_chunks)
        return f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {query}

Answer:"""

    def generate_answer(self, query: str) -> str:
        retrieved_chunks = self.retrieve(query)
        prompt = self.build_prompt(retrieved_chunks, query)
        response = self.generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]["generated_text"]
        return response.split("Answer:")[-1].strip(), retrieved_chunks
