# Local AI Starter - Qwen 3 + Ollama

A free local RAG and AI agent with Qwen 3 models using Ollama. This starter repo includes implementations for RAG (Retrieval-Augmented Generation) and AI Agents following the article: https://www.freecodecamp.org/news/build-a-local-ai/

## Features

- **RAG System**: Query PDF documents using local AI with HuggingFace Sentence Transformers embeddings
- **AI Agents**: Build intelligent agents with custom tools
- Fully local - no API keys or cloud services required
- Uses Qwen 3 models via Ollama for generation
- Uses Sentence Transformers for embeddings (automatically downloaded on first run)

## Prerequisites

- Python 3.8+
- Basic command-line familiarity
- Basic Python programming knowledge

## Quick Start

### 1. Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

**Windows:**
Download installer from https://ollama.com/download

### 2. Pull Qwen 3 Model

```bash
ollama pull qwen3:8b
```

Available models:
- `qwen3:0.6b` - Smallest, fastest
- `qwen3:4b` - Good balance for limited resources
- `qwen3:8b` - **Recommended** - Best balance for most systems
- `qwen3:14b`, `qwen3:32b` - Larger models for better performance

### 3. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Download Sample PDF (Optional)

For testing the RAG system:

```bash
wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
```

Or place your own PDF in the `data/` directory and update `PDF_FILENAME` in `rag_local.py`.

## Usage

### Start Ollama Server

In a separate terminal:

```bash
ollama serve
```

Keep this running while using the AI systems.

### RAG System

Query PDF documents with AI:

```bash
python rag_local.py
```

The script will:
1. Load your PDF from the `data/` directory
2. Split it into chunks
3. Create embeddings using Sentence Transformers (downloads ~90MB model on first run)
4. Store embeddings in ChromaDB
5. Run one sample query
6. Enter interactive mode where you can ask your own questions

**Note:** First run will download the `all-MiniLM-L6-v2` model automatically. This is a one-time download.

Once running, you can ask questions about your document. Type `quit` or `exit` to stop, or press Ctrl+C.

### AI Agent System

Run intelligent agents with custom tools:

```bash
python agent_local.py
```

The included agent can:
- Get current date/time in any format
- Use tools based on your queries

Extend by adding more `@tool` decorated functions in `agent_local.py`.

## Project Structure

```
local-ai-starter/
├── data/                  # Place your PDF documents here
├── chroma_db/            # Vector database (created automatically)
├── rag_local.py          # RAG implementation
├── agent_local.py        # AI Agent implementation
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Customization

### RAG System

Edit `rag_local.py` to:
- Change the PDF file: Update `PDF_FILENAME`
- Adjust chunk size: Modify `chunk_size` and `chunk_overlap` in `split_documents()`
- Change retrieval count: Update `k` value in `create_rag_chain()`
- Switch models: Change `llm_model_name` parameter
- Use different embeddings: Change `model_name` in `get_embedding_function()` (e.g., "all-mpnet-base-v2")

### AI Agents

Edit `agent_local.py` to:
- Add new tools: Create functions with `@tool` decorator
- Change model: Update `model_name` in `get_agent_llm()`
- Modify behavior: Adjust `temperature` parameter

### Advanced: Qwen 3 Thinking Mode

Append `/think` for step-by-step reasoning or `/no_think` for quick responses:

```python
response = llm.invoke("Solve 2x + 5 = 15 /think")
```

### Context Window

Increase for longer documents:

```python
llm = ChatOllama(model="qwen3:8b", num_ctx=16384)
```

## Troubleshooting

**Out of Memory:**
- Use a smaller model (`qwen3:4b`)
- Reduce `num_ctx` parameter
- Close other applications

**ChromaDB Issues:**
- Delete `chroma_db/` directory and re-run to rebuild index

**Ollama Connection Error:**
- Ensure `ollama serve` is running
- Check if another process is using the port

**PDF Loading Error:**
- Verify PDF file exists in `data/` directory
- Check `PDF_FILENAME` matches your file

## Resources

- [Qwen 3 GitHub](https://github.com/QwenLM/Qwen3)
- [Ollama Official Site](https://ollama.com/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Original Tutorial](https://www.freecodecamp.org/news/build-a-local-ai/)

## Demo

```python rag_local.py```
![](/demo/rag.png)
