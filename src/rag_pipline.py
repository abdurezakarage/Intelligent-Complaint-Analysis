from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch

class RAGPipeline:
    def __init__(
        self,
        vectorstore_path: str,
        embedding_model_name: str = "intfloat/e5-small-v2",
        llm_model_name: str = "tiiuae/falcon-7b-instruct",  # ✅ Open-access model
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

        self.generator = pipeline(
            "text-generation",
            model=llm_model_name,
            device=0 if device == "cuda" else -1,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7
        )

    def retrieve(self, query: str):
        return self.vectorstore.similarity_search(query, k=self.top_k)

    def build_prompt(self, context_chunks, question: str) -> str:
        max_context_length = 500
        context = "\n\n".join([doc.page_content[:max_context_length] for doc in context_chunks])
        prompt = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:
"""
        return prompt.strip()

    def generate_answer(self, question: str):
        context_chunks = self.retrieve(question)
        prompt = self.build_prompt(context_chunks, question)
        output = self.generator(prompt)[0]["generated_text"]
        answer = output.split("Answer:")[-1].strip()
        return answer, context_chunks