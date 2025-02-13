import os
import io
import base64
import streamlit as st
from PIL import Image
import pdf2image
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.colors import HexColor
import language_tool_python
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
import pytesseract
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize grammar checker using the remote server
tool = language_tool_python.LanguageToolPublicAPI('en-US')  # Use remote server

# Function to get response from Gemini model
def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-2.0-flash-001')
    response = model.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# Optimized function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        # Check if the file has content
        if uploaded_file.size == 0:
            raise ValueError("The uploaded file is empty.")
        
        # Use PyPDF2 for text extraction first
        reader = PdfReader(uploaded_file)
        extracted_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

        if not extracted_text.strip():
            # If no text is found, fall back to OCR using pytesseract (only if necessary)
            images = pdf2image.convert_from_bytes(uploaded_file.read())
            extracted_text = ""
            for image in images:
                extracted_text += pytesseract.image_to_string(image, lang='eng')

        return extracted_text.strip() or "No extractable text found."
    except Exception as e:
        raise ValueError(f"Error processing the PDF: {str(e)}")

# Function to convert PDF to image with added error handling
def input_pdf_setup(uploaded_file):
    try:
        # Check if the file has content
        if uploaded_file.size == 0:
            raise ValueError("The uploaded file is empty.")
        
        # Convert PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        if not images:
            raise ValueError("Unable to convert PDF to image. The file might be corrupted.")
        
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr).decode()}]
        return pdf_parts
    except Exception as e:
        raise ValueError(f"Error converting PDF to image: {str(e)}")

# Function to create a well-formatted PDF
def create_pdf(response):
    pdf_output = io.BytesIO()
    doc = SimpleDocTemplate(pdf_output, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']
    story = []

    story.append(Paragraph("ATS Evaluation Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(response.replace("\n", "<br />"), normal_style))

    doc.build(story)
    pdf_output.seek(0)
    return pdf_output

# Function to check grammar and spelling
def check_grammar(text):
    matches = tool.check(text)
    return len(matches), matches

# Function to generate a graph of grammar errors
def generate_error_graph(errors):
    error_types = {}
    for error in errors:
        rule_id = error.ruleId
        error_types[rule_id] = error_types.get(rule_id, 0) + 1

    labels = list(error_types.keys())
    values = list(error_types.values())

    fig, ax = plt.subplots()
    ax.barh(labels, values, color="skyblue")
    ax.set_xlabel("Count")
    ax.set_ylabel("Error Types")
    ax.set_title("Grammar Error Distribution")
    st.pyplot(fig)

# Streamlit UI
st.set_page_config(page_title="ATS Resume Expert", page_icon="📄", layout="wide")

# Add a sidebar for additional features
with st.sidebar:
    st.header("Resume Tips")
    st.markdown("- Ensure your resume is ATS-friendly.")
    st.markdown("- Use job-specific keywords.")
    st.markdown("- Avoid using tables or images for critical details.")

# Main UI Section
st.title("📄 ATS Tracking System")

# Job Description and Resume Upload Section
st.markdown("### Job Description and Resume Upload")
input_text = st.text_area("Enter the Job Description:", key="input", height=150)

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF Uploaded Successfully!")

# Buttons for different functionalities
col1, col2 = st.columns(2)

with col1:
    submit1 = st.button("Analyze Resume")
with col2:
    submit3 = st.button("Calculate Match Percentage")

# Prompt Templates
input_prompt1 = """
You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
Your task is to evaluate the resume against the provided job description. Provide the percentage match between the resume and job description. 
Start with the percentage, followed by missing keywords, and end with final thoughts.
"""

# Function to handle the results display and report download
def display_results(response, resume_text=None, job_description_text=None):
    st.subheader("ATS Evaluation Result")
    st.write(response)

    if resume_text:
        num_errors, errors = check_grammar(resume_text)
        st.write(f"Grammar and Spelling Errors Found: {num_errors}")
        if num_errors > 0:
            for error in errors[:5]:
                st.write(f"- {error.message} in {error.context}")
            generate_error_graph(errors)

    pdf_output = create_pdf(response)

    st.download_button(
        label="Download Report (PDF)",
        data=pdf_output,
        file_name="ats_evaluation_report.pdf",
        mime="application/pdf"
    )

# Submit Button Actions
if submit1:
    if uploaded_file is not None:
        with st.spinner("Processing your resume..."):
            try:
                # Process PDF only once
                pdf_content = input_pdf_setup(uploaded_file)
                extracted_text = extract_text_from_pdf(uploaded_file)

                # Analyze resume using Gemini API
                response = get_gemini_response(input_text, pdf_content, input_prompt1)
                display_results(response, resume_text=extracted_text)
            except Exception as e:
                st.error(str(e))
    else:
        st.error("Please upload the resume!")

elif submit3:
    if uploaded_file is not None:
        with st.spinner("Calculating match percentage..."):
            try:
                # Process PDF only once
                pdf_content = input_pdf_setup(uploaded_file)
                extracted_text = extract_text_from_pdf(uploaded_file)

                # Calculate match percentage using Gemini API
                response = get_gemini_response(input_text, pdf_content, input_prompt3)
                display_results(response, resume_text=extracted_text)
            except Exception as e:
                st.error(str(e))
    else:
        st.error("Please upload the resume!")