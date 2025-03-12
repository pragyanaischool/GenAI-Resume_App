import streamlit as st
import streamlit_ext as ste
import os
import json
from langchain.chat_models import ChatGroq
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader  # PDF text extraction

# Load Hugging Face embedding model
hf_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text from uploaded PDF
def extract_text_from_upload(uploaded_file):
    text = ""
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to get Groq API key & model selection
def get_groq_llm():
    groq_api_key = os.getenv("GROQ_API_KEY") or st.text_input("Enter Groq API Key:", type="password")
    model_name = st.selectbox("Select a model:", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"], index=0)
    
    if groq_api_key:
        return ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.1)
    else:
        st.error("Please enter a valid Groq API key.")
        return None

# Function to generate a resume using Groq LLM
def generate_resume(llm, extracted_text):
    embedded_text = hf_embedder.encode(extracted_text)  # Generate embeddings
    
    prompt = f"""
    You are an AI resume writer. Given the following extracted text from a document, generate a well-structured resume in JSON format.

    Extracted Text:
    {extracted_text}

    JSON Resume Format:
    {{
        "name": "John Doe",
        "email": "johndoe@email.com",
        "phone": "123-456-7890",
        "summary": "A brief professional summary",
        "education": [
            {{
                "degree": "Bachelor's in Computer Science",
                "institution": "XYZ University",
                "year": "2023"
            }}
        ],
        "experience": [
            {{
                "job_title": "Software Engineer",
                "company": "TechCorp",
                "years": "2020-2023",
                "responsibilities": ["Developed AI models", "Led a team of engineers"]
            }}
        ],
        "skills": ["Python", "Machine Learning", "Deep Learning"]
    }}

    Generate the resume based on the extracted information:
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# Streamlit UI
st.title("AI Resume Generator with Groq & Hugging Face Embeddings")

# Upload PDF file
uploaded_file = st.file_uploader("Upload your resume document (PDF)", type=["pdf"])

# Get LLM model
llm = get_groq_llm()

if uploaded_file and llm:
    with st.spinner("Extracting text..."):
        extracted_text = extract_text_from_upload(uploaded_file)

    if extracted_text:
        st.success("Text extracted successfully!")
        
        # Generate resume
        with st.spinner("Generating Resume..."):
            resume_json = generate_resume(llm, extracted_text)

        try:
            resume_data = json.loads(resume_json)
            st.json(resume_data)
        except json.JSONDecodeError:
            st.error("Error parsing JSON response. Please try again.")
