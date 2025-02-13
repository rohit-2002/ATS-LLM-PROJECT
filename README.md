# ATS Resume Expert

ATS Resume Expert is a Streamlit-based application designed to evaluate resumes against job descriptions. It uses advanced AI tools, such as Google Gemini API, grammar checking with LanguageTool, and OCR for text extraction, to provide insights and improve resume compatibility with ATS (Applicant Tracking Systems).

## Live Site

You can view the live version of this project at the following link:
[Live Website](https://ats-llm-project.streamlit.app/)
---
## Features

- **Resume Analysis**: Review resumes against job descriptions and highlight strengths and weaknesses.
- **Match Percentage Calculation**: Calculate the compatibility percentage between resumes and job descriptions.
- **Grammar Check**: Detect grammar and spelling issues in resumes and display a graphical distribution of errors.
- **PDF Generation**: Generate a professional evaluation report in PDF format.
- **OCR Support**: Extract text from image-based PDFs using OCR (pytesseract).
- **Streamlit UI**: Simple and interactive user interface for ease of use.

---

## Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or higher
- Virtual environment (optional, but recommended)

### Python Libraries

Install the required libraries using pip:

```bash
pip install -r requirements.txt
```
## Environment Variables

The app requires a `.env` file in the root directory to securely store API keys. Follow these steps to set it up:

1. Create a `.env` file in the project root directory.
2. Add your Google Gemini API key in the following format:

```plaintext
GOOGLE_API_KEY=your_api_key_here
```

## How to Run

Follow these steps to set up and run the application:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/ats-resume-expert.git
   ```
   
2. **Navigate to the project directory**:
   ```bash
   cd ATS LLM PROJECT
   ```

3. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
5. **Open the app in your browser**:
   Once the app is running, open your browser and go to http://localhost:8501.
