# GenAI-Lawgic

A FastAPI-powered legal document simplifier and chatbot. Upload legal PDFs,
process them, and ask questions for instant, plain-language answers.

## Features

- Upload and process legal PDF documents
- Ask questions and get simplified answers
- Modern, clean web UI

## Getting Started

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run the server

```
uvicorn main:app --reload
```

### 4. Access the app

Open your browser and go to:

```
http://localhost:8000/
```

## Usage

1. Upload your PDF document in the sidebar.
2. Click "Process Documents".
3. Ask your legal question in the chat.

## Deployment

- See `.gitignore` for recommended exclusions.
- You can deploy on platforms like Railway, Render, Heroku, or any cloud VM.

## License

MIT
# AI-Sprint-2.0
