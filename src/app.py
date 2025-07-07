
import gradio as gr
#from rag_pipeline import RAGPipeline  # Ensure this file contains your updated RAGPipeline class

# ✅ Initialize the RAG system
rag = RAGPipeline(vectorstore_path="/content/drive/MyDrive/ModelEmbeding")

# ✅ Chat function that returns the answer and source chunks
def chat(query):
    if not query.strip():
        return "", ["", "", ""]

    answer, sources = rag.generate_answer(query)
    source_texts = [doc.page_content[:500] for doc in sources[:3]]  # Limit to 500 chars, top 3 sources
    while len(source_texts) < 3:
        source_texts.append("")  # Pad with empty strings if < 3 sources

    return answer, source_texts[0], source_texts[1], source_texts[2]

# ✅ Gradio UI
with gr.Blocks(title="CrediTrust Complaint Assistant", theme=gr.themes.Base(primary_hue="indigo")) as demo:
    with gr.Column(scale=1):
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px;">
                <h1 style="color:#4F46E5;">💬 CrediTrust Complaint Chatbot</h1>
                <p style="font-size: 16px;">Ask questions about consumer financial complaints.<br>
                The AI responds using real complaint narratives retrieved from a trusted vector database.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., Why are customers unhappy with Buy Now Pay Later?",
                    lines=2
                )
                ask_button = gr.Button("Ask", variant="primary")

            with gr.Column(scale=6):
                answer_output = gr.Textbox(
                    label="Generated Answer",
                    lines=6,
                    interactive=False,
                    show_copy_button=True
                )

        gr.Markdown("### 📚 Retrieved Complaint Sources")

        # ✅ Create 3 collapsible boxes for source texts
        with gr.Accordion("Source 1", open=False):
            source1 = gr.Textbox(label="", interactive=False, lines=5)

        with gr.Accordion("Source 2", open=False):
            source2 = gr.Textbox(label="", interactive=False, lines=5)

        with gr.Accordion("Source 3", open=False):
            source3 = gr.Textbox(label="", interactive=False, lines=5)

        clear_button = gr.Button("Clear", variant="secondary")

    # ✅ Connect the buttons to functions
    ask_button.click(
        fn=chat,
        inputs=question_input,
        outputs=[answer_output, source1, source2, source3]
    )

    clear_button.click(
        fn=lambda: ("", "", "", "", ""),
        inputs=[],
        outputs=[answer_output, source1, source2, source3]
    )

# ✅ Launch the app
demo.launch()