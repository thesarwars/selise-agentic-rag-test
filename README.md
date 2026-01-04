# Mini Agentic RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with agentic capabilities including tool calling, self-reflection, and grounded answer generation.

## Features

✨ **Core Capabilities:**
- 📚 Document chunking and embedding generation
- 🗄️ Vector storage using ChromaDB
- 🔍 Intelligent retrieval with semantic search
- 🤖 Agentic behavior with tool calling
- 🧠 Self-reflection for answer quality
- ✅ Grounded answers to minimize hallucination

## Architecture

```
User Question
    ↓
Agent (GPT-4)
    ↓
Tool Calling → Retrieval Tool
    ↓
Vector Store (ChromaDB)
    ↓
Retrieved Documents
    ↓
Answer Generation
    ↓
Self-Reflection (Critic)
    ↓
Improved Answer (if needed)
    ↓
Final Response
```

## Installation

1. **Clone or download the project**

2. **Install dependencies:**
```bash
pip install openai chromadb
```

3. **Set up OpenAI API key:**
The system will prompt you for your API key when you run it, or set it as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Start

### 1. Add Your Documents

Place your knowledge base documents in the `documents/` directory:
- Supported formats: `.txt`, `.md`
- The system will automatically chunk and embed them

Sample documents are already provided:
- `python_basics.md`
- `machine_learning.md`
- `deep_learning.md`

### 2. Run the System

```bash
python main.py
```

Or specify a custom documents directory:
```bash
python main.py /path/to/your/documents
```

### 3. Chat with Your Knowledge Base

```
You: What are the main types of machine learning?
Assistant: Based on the retrieved documents, there are three main types of machine learning:

1. Supervised Learning - Learning from labeled data for classification and regression tasks
2. Unsupervised Learning - Learning from unlabeled data for clustering and dimensionality reduction
3. Reinforcement Learning - Learning through interaction with an environment using rewards

[Answer generated with retrieval-based grounding and self-reflection]
```

## Usage

### Interactive Chat Commands

- **Ask questions**: Simply type your question
- **Toggle verbose mode**: `verbose on` or `verbose off`
- **Clear conversation**: `clear`
- **Exit**: `quit` or `exit`

### Verbose Mode

Enable verbose mode to see the full agentic process:

```bash
You: verbose on
You: What is backpropagation?
```

Output will show:
1. Tool calling to retrieve documents
2. Retrieved document snippets
3. Initial answer generation
4. Self-reflection critic evaluation
5. Answer improvement (if needed)
6. Final response

## Project Structure

```
.
├── main.py                 # Main entry point and CLI
├── document_processor.py   # Document loading and chunking
├── vector_store.py        # ChromaDB integration and embeddings
├── retrieval_tool.py      # Retrieval tool for agent
├── agentic_rag.py         # Agent with self-reflection
├── documents/             # Knowledge base documents
│   ├── python_basics.md
│   ├── machine_learning.md
│   └── deep_learning.md
├── chroma_db/            # ChromaDB storage (auto-created)
└── README.md
```

## Components

### 1. Document Processor (`document_processor.py`)
- Loads documents from directory
- Chunks text intelligently at sentence boundaries
- Preserves metadata (source, filepath)
- Configurable chunk size and overlap

### 2. Vector Store (`vector_store.py`)
- Uses ChromaDB for persistent storage
- Generates embeddings with OpenAI `text-embedding-3-small`
- Cosine similarity search
- Batch processing for efficiency

### 3. Retrieval Tool (`retrieval_tool.py`)
- Implements OpenAI function calling interface
- Retrieves top-k relevant documents
- Formats results with source attribution

### 4. Agentic RAG (`agentic_rag.py`)
- **Tool Calling**: Automatically retrieves relevant documents
- **Self-Reflection**: Critic evaluates answer quality
- **Answer Improvement**: Regenerates answer if issues detected
- **Grounding**: All answers based on retrieved documents

### 5. Main Application (`main.py`)
- CLI interface for interaction
- Handles document indexing
- Manages conversation flow

## How It Works

### 1. Document Indexing
```python
# Load and chunk documents
chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
documents = chunker.load_documents_from_directory("./documents")
chunks = chunker.process_documents(documents)

# Generate embeddings and store
vector_store = VectorStore()
vector_store.add_documents(chunks)
```

### 2. Query Processing
```python
# User asks a question
question = "What is a neural network?"

# Agent automatically:
# 1. Calls retrieval tool
# 2. Gets relevant documents
# 3. Generates answer
# 4. Self-reflects on quality
# 5. Improves if needed

answer = agent.chat(question)
```

### 3. Self-Reflection Process
The critic evaluates answers on:
- **Accuracy**: Factually correct based on documents
- **Grounding**: Well-supported by retrieved content
- **Completeness**: Fully addresses the question
- **Hallucination Risk**: No unsupported claims

If confidence score < 7 or issues detected, the answer is improved.

## Configuration

### Chunk Size
Adjust in `main.py`:
```python
chunker = DocumentChunker(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50     # Overlap between chunks
)
```

### Retrieval Settings
Modify top_k in retrieval:
```python
# In retrieval_tool.py or agent calls
results = vector_store.retrieve(query, top_k=5)
```

### Model Selection
Change the LLM model in `agentic_rag.py`:
```python
agent = AgenticRAG(retrieval_tool, model="gpt-4o-mini")
# Options: gpt-4o-mini, gpt-4o, gpt-4-turbo, etc.
```

## Advanced Features

### Custom Documents
Add your own domain knowledge:
1. Create `.txt` or `.md` files
2. Place in `documents/` directory
3. Run the system - it will auto-index

### Reindexing
To reindex documents:
1. Run `python main.py`
2. When prompted, choose 'y' to clear and reindex

### Programmatic Usage
```python
from document_processor import DocumentChunker
from vector_store import VectorStore
from retrieval_tool import RetrievalTool
from agentic_rag import AgenticRAG

# Setup
vector_store = VectorStore()
retrieval_tool = RetrievalTool(vector_store)
agent = AgenticRAG(retrieval_tool)

# Get answer with full details
result = agent.answer_question(
    "What is machine learning?",
    use_reflection=True,
    verbose=True
)

print(result['answer'])
print(result['evaluation'])  # Critic feedback
```

## Minimizing Hallucination

The system uses multiple techniques:
1. **Retrieval-based grounding**: All answers cite sources
2. **Tool calling**: Agent must retrieve before answering
3. **Self-reflection**: Critic checks for unsupported claims
4. **Confidence scoring**: Low confidence triggers improvement
5. **Source attribution**: Documents are cited in answers

## Troubleshooting

### No documents found
- Ensure documents are in `./documents/` directory
- Check file extensions (`.txt` or `.md`)
- Verify read permissions

### OpenAI API errors
- Check API key is valid
- Ensure sufficient credits
- Check internet connection

### ChromaDB errors
- Delete `chroma_db/` directory and reindex
- Check write permissions

### Import errors
- Ensure all dependencies installed: `pip install openai chromadb`
- Check Python version (3.8+)

## Performance Tips

1. **Batch indexing**: Index all documents at once
2. **Chunk size**: Balance between context and specificity
3. **Top-k retrieval**: More documents = better context but slower
4. **Model choice**: GPT-4o-mini is faster and cheaper
5. **Verbose mode**: Disable for production use

## Future Enhancements

Potential improvements:
- [ ] Support for PDF, DOCX, HTML documents
- [ ] Multi-modal support (images, tables)
- [ ] Query expansion and rewriting
- [ ] Hybrid search (keyword + semantic)
- [ ] Web API with FastAPI
- [ ] Conversation memory
- [ ] Source highlighting
- [ ] Evaluation metrics dashboard

## License

MIT License - Feel free to use and modify

## Contributing

Contributions welcome! Areas for improvement:
- Additional document formats
- Better chunking strategies
- Alternative vector stores
- Enhanced reflection mechanisms
- UI/Web interface

## Credits

Built with:
- OpenAI GPT-4 for language understanding
- ChromaDB for vector storage
- Python for implementation# AgenticRag
# AgenticRag
# AgenticRag
# AgenticRag
# selise-agentic-rag-test
