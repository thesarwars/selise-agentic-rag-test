import os
from document_processor import DocumentChunker
from vector_store import VectorStore
from retrieval_tool import RetrievalTool
from agentic_rag import AgenticRAG
from dotenv import load_dotenv


def example_setup():
    print("=" * 60)
    print("EXAMPLE 1: Setting up the RAG system")
    print("=" * 60)
    
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    documents = chunker.load_documents_from_directory("./documents")
    print(f"Loaded {len(documents)} documents")
    
    chunks = chunker.process_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Use the same collection name as main.py
    vector_store = VectorStore(collection_name="knowledge_base")
    
    print("\n" + "=" * 60)
    response = input("Clear existing data and reindex all? (y/n): ").strip().lower()

    if response == 'y':
        print("\n4. Clearing existing collection...")
        vector_store.clear_collection()
        
        print("\n5. Adding new documents to vector store...")
        vector_store.add_documents(chunks)
        
        final_count = vector_store.count_documents()
        print(f"\n" + "=" * 60)
        print("✓ REINDEXING COMPLETE!")
        print("=" * 60)
        print(f"Documents indexed: {final_count}")
        # print("\nYou can now query about cricket!")
    else:
        print("\nCancelled. No changes made.")
    
    return vector_store


def example_retrieval(vector_store):
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Direct retrieval")
    print("=" * 60)
    
    query = "What is a neural network?"
    print(f"Query: {query}\n")
    
    results = vector_store.retrieve(query, top_k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"[Result {i}] Source: {doc['metadata']['source']}")
        print(f"Text: {doc['text'][:200]}...")
        print(f"Distance: {doc['distance']:.4f}\n")


def example_basic_agent(vector_store):
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Basic agent usage")
    print("=" * 60)
    
    retrieval_tool = RetrievalTool(vector_store)
    agent = AgenticRAG(retrieval_tool)
    
    question = "What are the main types of machine learning?"
    print(f"Question: {question}\n")
    
    answer = agent.chat(question, verbose=False)
    print(f"Answer: {answer}\n")


def example_verbose_mode(vector_store):
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Verbose mode (full agentic process)")
    print("=" * 60)
    
    retrieval_tool = RetrievalTool(vector_store)
    agent = AgenticRAG(retrieval_tool)
    
    question = "Explain backpropagation in neural networks"
    
    result = agent.answer_question(question, use_reflection=True, verbose=True)
    
    print("\n" + "-" * 60)
    print("RESULT DETAILS:")
    print("-" * 60)
    print(f"Reflection used: {result['reflection_used']}")
    if result['evaluation']:
        print(f"Confidence score: {result['evaluation'].get('confidence_score')}/10")
        print(f"Is grounded: {result['evaluation'].get('is_grounded')}")


def example_multiple_questions(vector_store):
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Multiple questions")
    print("=" * 60)
    
    retrieval_tool = RetrievalTool(vector_store)
    agent = AgenticRAG(retrieval_tool)
    
    questions = [
        "What is overfitting in machine learning?",
        "How can I prevent overfitting?",
        "What's the difference between supervised and unsupervised learning?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        answer = agent.chat(question, verbose=False)
        print(f"Answer: {answer}\n")
        print("-" * 60)


def example_custom_retrieval(vector_store):
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Custom retrieval settings")
    print("=" * 60)
    
    retrieval_tool = RetrievalTool(vector_store)
    
    query = "When were the modern Laws of cricket first written down and printed?"
    print(f"Query: {query}")
    print(f"Retrieving top 4 documents...\n")
    
    results = retrieval_tool.retrieve(query, top_k=4)
    print(results)


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    if not os.getenv("AZURE_OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print("Setting up vector store...")
    vector_store = example_setup()
    
    #here I can run any function that I preseted above
    
    # example_retrieval(vector_store)
    # example_basic_agent(vector_store)
    # example_verbose_mode(vector_store)
    example_multiple_questions(vector_store)
    # example_custom_retrieval(vector_store)
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
