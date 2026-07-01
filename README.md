# Course Study Bot

A small RAG (Retrieval-Augmented Generation) chatbot that answers questions **only**
from your friend's course PDFs, and can generate revision notes and likely exam
questions from them.

## How it works
1. `ingest.py` reads every PDF in `pdfs/`, splits them into chunks, and turns each
   chunk into a vector using a **free local embedding model** (no API cost, no
   internet needed for this step). These are stored in a local database (ChromaDB).
2. `app.py` is the Streamlit chat app. When a question is asked, it finds the most
   relevant chunks from the database and sends *only those* to Gemini, with strict
   instructions to answer only from that context — otherwise it says the topic isn't
   covered.

## One-time setup

### 1. Get a free Gemini API key
This is **different from your Google One subscription** — it's a separate free
developer key. Go to https://aistudio.google.com/apikey, sign in, and create a key.
The free tier has daily rate limits but is plenty for personal study use.

### 2. Install dependencies
```bash
cd study-bot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key
```bash
cp .env.example .env
```
Open `.env` and paste your key:
```
GEMINI_API_KEY=your_actual_key_here
```

### 4. Add the PDFs
Put all 50-60 course PDFs into the `pdfs/` folder.

### 5. Build the database
```bash
python ingest.py
```
This only needs to be re-run when you add or remove PDFs. First run will download
a small (~90MB) embedding model automatically.

### 6. Run the app
```bash
streamlit run app.py
```
It'll open in your browser at `http://localhost:8501`.

## Using it
- **Ask a Question** — normal Q&A, grounded strictly in the PDFs, with source
  citations (filename + page number).
- **Revision Notes** — type a topic/chapter name, get structured study notes.
- **Exam Q&A Generator** — type a topic, choose how many questions, get likely
  exam questions with model answers.

## Notes & things you may want to tune
- **Scanned PDFs**: if a PDF is actually scanned images (no selectable text),
  `ingest.py` will warn you and skip it — it would need OCR, which isn't included
  yet but can be added (e.g. with `pytesseract`) if needed.
- **"Refuses too much / answers things it shouldn't"**: open `app.py` and adjust
  `DISTANCE_THRESHOLD` near the top. Lower = stricter, higher = more lenient. This
  needs a bit of trial and error with real questions from the PDFs.
- **Model name**: `MODEL_NAME = "gemini-2.5-flash"` in `app.py`. If Google renames
  or deprecates it by the time you deploy, you'll get a clear error — just swap in
  whatever current fast Gemini model name is available in your account.
- **Chunk size**: in `ingest.py`, `chunk_size=1000, overlap=150` (characters). Fine
  for most textbook-style PDFs; you can tune it if retrieval quality feels off.

## Deploying later (when you're ready)
This runs fully locally as-is. When you want your friend to access it remotely,
common free/cheap options: Streamlit Community Cloud, a small VM, or Hugging Face
Spaces. The `chroma_db/` folder and `pdfs/` folder just need to travel with the app.
