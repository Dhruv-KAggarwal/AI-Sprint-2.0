import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Setup
# -------------------------
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For hackathon demo, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Progress tracker
progress_data = {}


# -------------------------
# Core functions
# -------------------------
def get_pdf_text(pdf_file):
    """Extract text from uploaded PDF"""
    text = ""
    pdf_reader = PdfReader(pdf_file.file)
    for page in tqdm(pdf_reader.pages, desc="Extracting text", unit="page"):
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks, task_id):
    """Create a new FAISS index with HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    # Limit embeddings to first N chunks to save API quota
    max_chunks = 200
    text_chunks = text_chunks[:max_chunks]

    vector_store = FAISS.from_texts(
        tqdm(text_chunks, desc="Embedding chunks", unit="chunk"),
        embedding=embeddings
    )
    vector_store.save_local("faiss_index")
    progress_data[task_id] = {
        "status": "done",
        "progress": 100,
        "message": "Indexing complete ✅"
    }


def get_conversational_chain():
    """Build simplification chain"""
    prompt_template = """
You are a Compliance Review Agent specializing in Insurance Contracts. 
Your role is to help legal, compliance, and underwriting teams analyze insurance contracts, policies, and regulatory filings efficiently and accurately.

Guidelines:
- Extract and categorize clauses into insurance-specific legal types (e.g., coverage terms, exclusions, claims obligations, premium adjustments).
- Detect and prioritize regulatory and contractual risks.
- Align findings with insurance regulations (e.g., IRDAI, GDPR, HIPAA).
- Provide traceable, explainable, and auditable rationales for all outputs.
- If a requested answer is not found in the provided context, respond only with:
  "Answer is not available in the context."
- Do not fabricate information outside the context.

Context:
{context}

Question:
{question}

Expected Deliverables:
- Clause Inventory & Categorization
- Risk & Compliance Analysis (Aligned / Partial / Gap)
- Compliance Heatmap & Executive Summary

Answer:
"""

    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# -------------------------
# Routes
# -------------------------
@app.get("/")
def read_index():
    """Serve frontend index.html"""
    return FileResponse("static/index.html")


@app.post("/upload-pdf/")
async def upload_pdf(pdf: UploadFile, background_tasks: BackgroundTasks):
    """Upload PDF, extract text, create FAISS index in background"""
    task_id = str(uuid.uuid4())
    progress_data[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting PDF processing..."
    }

    text = get_pdf_text(pdf)
    chunks = get_text_chunks(text)
    progress_data[task_id] = {
        "status": "processing",
        "progress": 50,
        "message": "Embedding & indexing..."
    }
    background_tasks.add_task(get_vector_store, chunks, task_id)

    return {"task_id": task_id, "status": "⚡ PDF uploaded, processing in background..."}

@app.get("/progress/")
async def get_progress(task_id: str):
    """Fetch current progress for a given task"""
    return progress_data.get(
        task_id,
        {"status": "unknown", "progress": 0, "message": "Task not found"}
    )


@app.post("/simplify/")
async def simplify(question: str = Form(...)):
    """Simplify legal text based on PDF context"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "faiss_index", embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(question, k=3)  # Limit retrieval

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )
    return {"answer": response["output_text"]}


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT automatically
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
