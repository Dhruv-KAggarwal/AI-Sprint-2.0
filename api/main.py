from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PyPDF2 import PdfReader
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# FastAPI App Setup
# =========================================================
app = FastAPI()

# Allow frontend requests (all origins for hackathon/demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================================================
# Core Utilities
# =========================================================
def get_pdf_text(pdf_file):
    """Extract raw text from uploaded PDF"""
    text = ""
    pdf_reader = PdfReader(pdf_file.file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def get_text_chunks(text):
    """Split text into overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def get_vector_store(text_chunks, clear_old: bool = False):
    """Create FAISS index with HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Clear old index if requested
    if clear_old and os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


# =========================================================
# Prompt Chains
# =========================================================
def get_simplifier_chain():
    """Chain for Legal Document Simplifier AI"""
    prompt_template = """
You are a Legal Document Simplifier AI. 
Your task is to take complex legal text and explain it in clear, plain, and accurate language
that a non-lawyer can easily understand.

Guidelines:
- Use simple words, short sentences, and examples.
- Do NOT change meaning or remove important details.
- If an answer is not present in the context, reply: "answer is not available in the context".
- Never fabricate or assume outside information.

Context:
{context}

Question:
{question}

Simplified Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def get_compliance_chain():
    """Chain for AI Compliance Review Agent (Insurance Contracts)"""
    prompt_template = """
You are an AI Compliance Review Agent designed for the insurance sector.
You will analyze contracts, policies, and regulatory filings to:
- Extract & categorize insurance-specific clauses (coverage terms, exclusions, claims obligations, premium adjustments).
- Detect and prioritize risks by financial exposure, regulatory impact, and urgency.
- Map clauses to compliance frameworks (IRDAI, GDPR, HIPAA, internal policies).
- Provide explainable and auditable reasoning for each risk.

Guidelines:
- Preserve original meaning. Do NOT invent facts.
- If evidence is missing in context, reply: "Not Supported by Context".
- Rank risks using: combined_score = (financial_exposure*0.5 + regulatory_impact*0.3 + urgency*0.2).
- Provide outputs in two forms:
   1. JSON_OUTPUT: structured clauses, risks, compliance status (Aligned / Partial / Gap).
   2. HUMAN_READABLE: plain-language explanation with executive summary.

End every answer with:
"This analysis is for informational purposes only and does not constitute legal advice."

Context:
{context}

Question:
{question}

Compliance Review Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# =========================================================
# API Routes
# =========================================================
@app.get("/")
def read_index():
    """Serve frontend index.html"""
    return FileResponse("static/index.html")


@app.post("/upload-pdf/")
async def upload_pdf(pdf: UploadFile, clear_old: bool = Form(False)):
    """
    Upload PDF, extract text, and build FAISS index.
    User can choose to clear old FAISS index with clear_old=True.
    """
    text = get_pdf_text(pdf)
    chunks = get_text_chunks(text)
    get_vector_store(chunks, clear_old=clear_old)
    return {
        "status": "✅ PDF processed and indexed successfully",
        "cleared_old": clear_old
    }


@app.post("/simplify/")
async def simplify(question: str = Form(...)):
    """Simplify legal text using the Simplifier AI"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question)

    chain = get_simplifier_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return {"answer": response["output_text"]}


@app.post("/compliance/")
async def compliance(question: str = Form(...)):
    """Analyze compliance & risks using Compliance Review Agent"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question)

    chain = get_compliance_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return {"answer": response["output_text"]}


# =========================================================
# Run App
# =========================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
