# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

class RAGPipeline:
    def __init__(self, vectorstore_path: str):
        self.vectorstore_path = vectorstore_path
        # Dummy documents simulating retrieved complaints
        self.documents = [
            {
                "text": "Many customers report that Buy Now Pay Later services lead to confusion over repayment terms, unexpected fees, and credit score impact.",
                "metadata": {"product": "Buy Now Pay Later", "source": "complaint_1"}
            },
            {
                "text": "Users have complained about unclear payment schedules and difficulty disputing incorrect charges with Buy Now Pay Later providers.",
                "metadata": {"product": "Buy Now Pay Later", "source": "complaint_2"}
            }
        ]

        model_id = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def retrieve_documents(self, query: str) -> List[dict]:
        return self.documents[:2]

    def generate_answer(self, query: str):
        docs = self.retrieve_documents(query)
        context = "\n".join(doc["text"] for doc in docs)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer, docs

rag = RAGPipeline(
    vectorstore_path="/content/drive/MyDrive/Embeding",
)

query = "Why are customers unhappy with Buy Now Pay Later?"
answer, sources = rag.generate_answer(query)

print("🧠 Answer:\n", answer)

print("\n📚 Retrieved Source Example:")
for i, doc in enumerate(sources[:2]):
    print(f"\n--- Source {i+1} ---")
    print(doc["text"][:300] + "...")

# task 4 in gradio
import gradio as gr

# ✅ Initialize the RAG system
rag = RAGPipeline(vectorstore_path="/content/drive/MyDrive/Embeding")

# ✅ Chat function with history
def chat(query, history):
    if not query.strip():
        return gr.update(value=""), history, "", "", ""

    answer, sources = rag.generate_answer(query)
    history.append((query, answer))

    source_texts = [doc.get("text", "")[:500] for doc in sources[:3]]
    while len(source_texts) < 3:
        source_texts.append("")

    return gr.update(value=""), history, source_texts[0], source_texts[1], source_texts[2]

# ✅ Clear function
def clear_all():
    return "", [], "", "", ""

# ✅ Gradio UI
with gr.Blocks(title="CrediTrust Complaint Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1 style="color:#4F46E5;"> CrediTrust Complaint Chatbot</h1>
        </div>
        """,
        elem_id="header"
    )

    chatbot = gr.Chatbot(label="Conversation History", height=400)

    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., Why are customers unhappy with Buy Now Pay Later?",
                lines=2
            )
            ask_button = gr.Button("Ask", variant="primary")
            clear_button = gr.Button("Clear All", variant="secondary")

        with gr.Column(scale=6):
            gr.Markdown("### 📚 Retrieved Complaint Sources")
            with gr.Accordion("Source 1", open=False):
                source1 = gr.Textbox(label="", interactive=False, lines=4)
            with gr.Accordion("Source 2", open=False):
                source2 = gr.Textbox(label="", interactive=False, lines=4)
            with gr.Accordion("Source 3", open=False):
                source3 = gr.Textbox(label="", interactive=False, lines=4)

    # ✅ State to store chat history
    history_state = gr.State([])

    # ✅ Connect buttons
    ask_button.click(
        fn=chat,
        inputs=[question_input, history_state],
        outputs=[question_input, chatbot, source1, source2, source3]
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[question_input, chatbot, source1, source2, source3, history_state]
    )

# ✅ Launch the app
demo.launch()
