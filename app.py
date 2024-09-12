import streamlit as st
from models.model import initialize_llm, initialize_embeddings, initialize_vectorstore, create_rag_chain
from langchain_community.document_loaders import PyPDFLoader
import tempfile

# Hardcoded API Key and PDF path
GROQ_API_KEY = "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"
PDF_FILE_PATH = "data/dataset_informasi_2.pdf"

def initialize_rag_model():
    try:
        llm = initialize_llm(GROQ_API_KEY)
        embeddings = initialize_embeddings()
        st.session_state["llm"] = llm
        st.session_state["embeddings"] = embeddings
        st.success("API key set and model initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize model: {e}")

def process_documents():
    try:
        with open(PDF_FILE_PATH, "rb") as file:
            pdf_loader = PyPDFLoader(PDF_FILE_PATH)
            documents = pdf_loader.load()
        
        if documents:
            if "retriever" not in st.session_state:
                retriever = initialize_vectorstore(documents, st.session_state["embeddings"])
                st.session_state["retriever"] = retriever
                st.session_state["documents_loaded"] = True
                st.write("Documents processed successfully.")
            else:
                st.write("Documents have already been processed.")
        else:
            st.write("No valid documents found in the file.")
    except Exception as e:
        st.error(f"Failed to process documents: {e}")

def handle_query():
    input_text = st.text_area("Mari bertanya")
    if st.button("Dapatkan Jawaban"):
        if input_text:
            try:
                if "retriever" in st.session_state and "llm" in st.session_state:
                    rag_chain = create_rag_chain(st.session_state["retriever"], st.session_state["llm"])
                    response = rag_chain.invoke({"input": input_text})
                    st.write("### Answer")
                    st.write(response['answer'])
                else:
                    st.error("Model or retriever is not initialized.")
            except Exception as e:
                st.error(f"Failed to retrieve answer: {e}")
        else:
            st.write("Mari bertanya")

def main():
    st.title("Wisata Chatbot AI! with LLAMA3🚀")
    st.write("Tanyakan apa saja kepada Chatbot kami.")
    st.markdown("")

    initialize_rag_model()
    process_documents()
    handle_query()

if __name__ == "__main__":
    main()
