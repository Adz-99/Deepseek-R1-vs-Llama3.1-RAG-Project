import os

import streamlit as st

from RAG_utils import process_document, get_answer

working_dir = os.getcwd()

# Set up streamlit page and config
st.set_page_config(
    page_title="Deepseek-R1 vs Llama3",
    page_icon="🤖",
    layout="centered"
)

st.title("🐳 Deepseek-R1 vs LLama3 🦙")

uploaded_file = st.file_uploader("Upload a PDF file for RAG", type=["pdf"])

if uploaded_file is not None:
    # Create a file path to save the document to
    file_path = os.path.join(working_dir, uploaded_file.name)
    # Save the file to working directory
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    processed_document = process_document(uploaded_file.name)
    st.success("File uploaded successfully")

# Setup text prompt to allow user to ask a question
user_prompt = st.text_area("Ask a question about the document...")

if st.button("Submit"):
    # Once user asks a question, send to both models
    answers = get_answer(user_prompt)
    deepseek_answer = answers["deepseek_answer"]
    llama_answer = answers["llama_answer"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Deepseek-R1 Response")
        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #090909;">
                {deepseek_answer}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### Llama3 Response")
        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #090909;">
                {llama_answer}
            </div>
            """,
            unsafe_allow_html=True
        )