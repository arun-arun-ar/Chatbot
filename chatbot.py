import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
from datetime import datetime
import dateparser

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state for appointments
if 'appointments' not in st.session_state:
    st.session_state.appointments = []

def get_pdf_text(pdf_docs):
    """Extracts text from the uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                text += page_text.encode('utf-8', 'ignore').decode('utf-8')
            except Exception as e:
                st.warning(f"Error extracting text from a page: {e}")
    return text

def get_text_chunks(text):
    """Splits the extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generates vector embeddings and stores them in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Creates a conversational chain with a custom prompt template."""
    prompt_template = """
    Provide an accurate and concise answer to the question based on the given context. 
    If the context does not contain relevant information to answer the question, respond with, 
    "The required information is not available in the provided context."
    Avoid making assumptions or providing fabricated information.

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Handles user input and generates a response."""
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Jarvis: ", response["output_text"])

def extract_date(query):
    """Extracts date from the query and converts it to YYYY-MM-DD format."""
    # Attempt to parse the date using the `dateparser`
    try:
        date = dateparser.parse(query)
        if date:
            return date.strftime("%Y-%m-%d")
        else:
            st.warning("Could not extract a valid date from your input. Please try a more specific format like 'Next Monday' or '2024-12-15'.")
            return None
    except Exception as e:
        st.warning(f"Error while extracting date: {str(e)}")
        return None

def validate_email(email):
    """Validates email format."""
    regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(regex, email) is not None

def validate_phone(phone):
    """Validates phone number format."""
    regex = r'^\+?[1-9]\d{1,14}$'
    return re.match(regex, phone) is not None

def book_appointment():
    """Collect user information and validate inputs."""
    with st.form(key='appointment_form'):
        name = st.text_input("What is your name?")
        phone = st.text_input("What is your phone number?")
        email = st.text_input("What is your email?")
        date = st.date_input("Select appointment date:")

        submit_button = st.form_submit_button(label='Submit Appointment')

        if submit_button:
            if not name:
                st.warning("Name is required.")
            elif not phone or not validate_phone(phone):
                st.warning("Please provide a valid phone number.")
            elif not email or not validate_email(email):
                st.warning("Please provide a valid email address.")
            elif not date:
                st.warning("Please select a date for the appointment.")
            else:
                appointment_details = {
                    "name": name,
                    "phone": phone,
                    "email": email,
                    "date": date.strftime("%Y-%m-%d")
                }
                st.session_state.appointments.append(appointment_details)
                st.success(f"Appointment confirmed for {name} at {phone} and {email} on {date.strftime('%Y-%m-%d')}.")

def show_appointments():
    """Displays the list of scheduled appointments."""
    if st.session_state.appointments:
        st.subheader("Scheduled Appointments:")
        for appointment in st.session_state.appointments:
            st.write(f"{appointment['name']} ({appointment['phone']}, {appointment['email']}) - {appointment['date']}")
    else:
        st.write("No appointments scheduled yet.")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with multiple PDFs using Gemini")

    user_question = st.text_input("Ask a question to Gemini from the PDF(s) you uploaded:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Simply upload your PDF files and press the 'Submit & Process' button to continue.",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Completed!")
                except Exception as e:
                    st.error(f"An error occurred while processing the PDFs: {e}")

        # Add functionality to collect user information when they ask to book an appointment
        book_appointment()

        # Display all scheduled appointments
        show_appointments()

if __name__ == "__main__":
    main()
