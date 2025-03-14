import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import re
import spacy


# Set page configuration
st.set_page_config(page_title="Project Prism")

# Constants
PDF_PATH = "databaseforrag.pdf"  # Update to your actual PDF path
VECTOR_STORE_DIR = "faiss_index"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
API_KEYS = [
    "AIzaSyDtYsVCQdw6qUni4XXUo_W5cW0HWueqVAU",
    "AIzaSyDHYjkvvK7sje7dUfnt4OIXcK0dhjZUN1A",
    "AIzaSyCnN3Bwnj5IymSn9PbUiHE4nUW0K8uzrPM"
]  # Replace with your actual API keys

# Load Spacy model for advanced text processing
nlp = spacy.load("en_core_web_sm")

# Globals
api_key_index = 0

def get_current_api_key():
    """Retrieve the current API key and rotate to the next if needed."""
    global api_key_index
    api_key = API_KEYS[api_key_index]
    api_key_index = (api_key_index + 1) % len(API_KEYS)
    return api_key

def extract_text_using_pytesseract(pdf_path):
    """Use OCR to extract text from images in the PDF using Tesseract."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text

def extract_text_using_pdfplumber(pdf_path):
    """Extract text from PDF using pdfplumber (handles tables and layouts better)."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def get_pdf_text(pdf_path):
    """Try multiple methods to extract text from a PDF."""
    try:
        # Use PyMuPDF to extract text (more accurate)
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")  # Get all text including hidden text
        # If PyMuPDF returns empty or incomplete, try pdfplumber
        if not text.strip():
            st.warning("PyMuPDF extraction returned no text, using pdfplumber as a fallback.")
            text = extract_text_using_pdfplumber(pdf_path)
        # If pdfplumber fails, use OCR (for scanned PDFs)
        if not text.strip():
            st.warning("No text extracted using pdfplumber, using OCR fallback.")
            text = extract_text_using_pytesseract(pdf_path)
        return text.strip()  # Return cleaned text
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        return ""

def clean_text(text):
    """Clean the text for better chunking and indexing."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = text.strip()
    return text

def get_text_chunks(text):
    """Split text into smaller, manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def get_vector_store(text_chunks=None):
    """Create or load the FAISS vector store."""
    index_file = os.path.join(VECTOR_STORE_DIR, "index.faiss")

    # If FAISS index file exists, try to load it
    if os.path.exists(index_file):
        st.info("Loading existing vector store...")
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=get_current_api_key()
            )
            vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return None
    else:
        # If index doesn't exist, create a new one
        if text_chunks:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", google_api_key=get_current_api_key()
                )
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
                vector_store.save_local(VECTOR_STORE_DIR)

                st.success("Vector store created successfully!")
                return vector_store
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                return None
        else:
            st.error("No text chunks provided, cannot create vector store.")
            return None

def get_conversational_chain():
    """Set up the conversational chain with prompt and model."""
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context.
        If the answer is not in the context, say, "Answer is not available in the context."

        Context: {context}
        Question: {question}

        Answer:
        """
        model = ChatGoogleGenerativeAI(
            model="gemini-pro", temperature=0.3, google_api_key=get_current_api_key()
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Error setting up conversational chain: {e}")
        return None

def process_user_query(user_question, vector_store):
    """Handle user query and generate response."""
    try:
        if not vector_store:
            st.error("Vector store is not available. Please process the document first.")
            return

        # Use the user's question to perform semantic search
        docs = vector_store.similarity_search(user_question, k=5)
        if docs:
            st.info(f"Found {len(docs)} related document chunks.")
        else:
            st.warning("No related documents found.")

        # Process the documents through the question-answering chain
        chain = get_conversational_chain()
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("**Reply:**", response["output_text"])
    except Exception as e:
        st.error(f"Error processing query: {e}")

def main():
    """Main Streamlit application."""
    st.markdown("""
    ## <h2>LOYOLA ACADEMY MCA MAJOR Projects List</h2>

    <h4>Every great mind has always started with an idea</h4>
    <h5>    What made them stand out from the rest?</h5>
    <h5>Their ideas were unique, bold, and visionary</h5>
    <h3>Dive into the club of great minds,
    where innovation thrives, and creativity knows no bounds</h3>
    <h2>Think uniquely, build uniquely, and inspire the world.</h2>
    """, unsafe_allow_html=True)

    # Add custom CSS styling for enhanced UI
    st.markdown("""
        <style>
        body {
            background-color: #eef2f9;
            font-family: 'Arial', sans-serif;
        }
        h1, h2, h3 {
            color: #2C3E50;
        }
        .stButton>button {
            background: linear-gradient(90deg, #2980b9, #9b59b6);
            color: white;
            border-radius: 8px;
            font-weight: bold;
            padding: 12px 24px;
            font-size: 16px;
            transition: background 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #3498db, #8e44ad);
        }
        .stTextInput>div>input {
            border-radius: 5px;
            border: 1px solid #ccd1d9;
            padding: 12px;
            font-size: 16px;
            background-color: #f7f9fc;
        }
        .stTextInput>div>input:focus {
            outline: none;
            border: 2px solid #2980b9;
            box-shadow: 0 0 5px rgba(52, 152, 219, 0.5);
        }
        .section-title {
            font-size: 24px;
            color: #34495e;
            font-weight: bold;
        }
        .stFileUploader>div {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Global variable to hold the vector store
    vector_store = None

    # User inputs
    user_question = st.text_input("Ask a question about the document")

    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            raw_text = get_pdf_text(PDF_PATH)
            if raw_text:
                cleaned_text = clean_text(raw_text)  # Clean the text
                text_chunks = get_text_chunks(cleaned_text)  # Create text chunks
                vector_store = get_vector_store(text_chunks)  # Only create if document is processed
                if vector_store:
                    st.success("Document processed successfully!")

    if user_question and vector_store:
        process_user_query(user_question, vector_store)

if __name__ == '__main__':
    main()
