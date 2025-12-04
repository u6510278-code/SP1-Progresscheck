# NotebookLM Clone (RAG Demo)

Simple RAG (Retrieval-Augmented Generation) app that:
- Ingests PDFs
- Stores embeddings in PostgreSQL with pgvector
- Answers questions using an LLM via OpenRouter

## Clone repo

git clone https://github.com/Ivank100/SP1.git
cd SP1

## Create Enviroment 

python3.12 -m venv env312
source env312/bin/activate (activate.fish if using fish shell)

## Install dependecies
pip install --upgrade pip
pip install -r requirements.txt


## Create .env file in root folder
Example code:
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=notebooklm
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
PGVECTOR_DIM=1536

DEEPSEEK_API_KEY=your-real-api-key-here
DEEPSEEK_BASE_URL=https://openrouter.ai/api/v1

## Start Postgres with Docker
docker compose up -d
docker ps (To check if its running)


## HOW TO USE

Download any PDF or MP3 file
Activate your enviroment with: source env312/bin/activate 
Ingest a PDF: python -m src.rag_index path/to/your_document.pdf

You should see:
[INFO] Reading PDF: path/to/your_document.pdf
[INFO] Extracted 36783 characters of text from PDF: path/to/your_document.pdf
[INFO] Created 29 chunks
[INFO] Embedding 29 chunks (pure Python, very light)...
[SUCCESS] Ingested pdf path/to/your_document.pdf as doc_id=... , 29 chunks

Copy the doc_id and paste it into testnotebooklm.py

do the same for audio 

then run python testnotebooklm.py

Ask questions to chatbot about your pdf or audio and enjoy










