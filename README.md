# RAG Document Q&A Chatbot
## Features
## Tech Stack
## Architecture

## Architecture

```text
┌──────────────┐
│  Upload PDF  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Extract Text │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Text Chunks  │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ Gemini Embeddings   │
└──────┬──────────────┘
       │
       ▼
┌──────────────┐
│    FAISS     │
│ Vector Store │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ User Query   │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ Query Embedding     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Similarity Search   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Relevant Chunks     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Gemini 2.5 Flash    │
│ Answer Generation   │
└──────┬──────────────┘
       │
       ▼
┌──────────────┐
│ Final Answer │
└──────────────┘
```
## Installation
## Usage
## Future Improvements

