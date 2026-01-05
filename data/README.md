# Data Directory

Place your PDF documents in this directory for use with the RAG system.

## Quick Start

Download a sample PDF for testing:

```bash
wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
```

Or on Windows/macOS without wget, download manually from:
https://arxiv.org/pdf/2307.09288.pdf

## Using Your Own PDFs

1. Copy your PDF file into this directory
2. Update the `PDF_FILENAME` variable in `rag_local.py` to match your filename
3. Run `python rag_local.py`

That's it!
