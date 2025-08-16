# vector_db

A minimal utility to build a vector database from PDF documents using
FAISS and sentence-transformers.

## Installation

```bash
pip install -r requirements.txt
```

## Example

```python
from vector_db import PDFVectorDB

db = PDFVectorDB()
db.build_from_pdf("document.pdf")
db.save("document_index")

# Later, load and query
other = PDFVectorDB()
other.load("document_index")
print(other.query("What is the topic?", top_k=3))
```
