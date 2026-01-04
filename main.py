import os
import getpass
import sys
from dotenv import load_dotenv
from document_processor import DocumentChunker
from vector_store import VectorStore
from retrieval_tool import RetrievalTool
from agentic_rag import AgenticRAG


def setup_knowledge_base(documents_dir: str = "./documents"):
    print("=" * 60)
    print("Setting up Agentic RAG System")
    print("=" * 60)
    print("\n1. Loading and chunking documents...")
    # Here I splitted the pdf into chunk file for easily upload to the model to avoid any interruption during the embed.
    chunker = DocumentChunker(chunk_size=1000)
    documents = chunker.load_documents_from_directory(documents_dir)
    
    if not documents:
        print(f"Warning: No documents found in {documents_dir}")
        print("Please add pdf, doc, .txt or .md files to the documents directory")
        return None
    
    print(f"Loaded {len(documents)} documents")
    
    chunks = chunker.process_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("\n2. Initializing vector store...")
    vector_store = VectorStore(collection_name="knowledge_base")
    
    existing_count = vector_store.count_documents()
    if existing_count > 0:
        print(f"Found {existing_count} existing documents")
        response = input("Clear existing documents and reindex? (y/n): ").lower()
        if response == 'y':
            vector_store.clear_collection()
            print("Cleared existing collection")
        else:
            print("Using existing documents")
            chunks = []
    
    if chunks:
        print(f"\n3. Generating embeddings and storing in ChromaDB...")
        vector_store.add_documents(chunks)
    
    print("\n4. Initializing agentic RAG system...")
    retrieval_tool = RetrievalTool(vector_store)
    agent = AgenticRAG(retrieval_tool)
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"Documents indexed: {vector_store.count_documents()}")
    print("Agent features: Tool Calling, Self-Reflection, Grounded Answers")
    return agent


def chat_loop(agent: AgenticRAG):
    print("\n" + "=" * 60)
    print("Agentic RAG Chat Interface")
    print("=" * 60)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'verbose on/off' - Toggle verbose mode")
    print("  'clear' - Clear conversation history")
    print("=" * 60 + "\n")
    
    verbose = False
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'clear':
                agent.reset_conversation()
                print("Conversation history cleared.\n")
                continue
            
            if user_input.lower().startswith('verbose'):
                if 'on' in user_input.lower():
                    verbose = True
                    print("Verbose mode enabled.\n")
                elif 'off' in user_input.lower():
                    verbose = False
                    print("Verbose mode disabled.\n")
                continue
            
            response = agent.chat(user_input, verbose=verbose)
            
            if not verbose:
                print(f"\nAssistant: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("No API key found. Choose your provider:")
        print("1. Standard OpenAI")
        print("2. Azure OpenAI")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            print("\nAzure OpenAI Configuration:")
            os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter your Azure OpenAI API Key: ")
            os.environ["AZURE_OPENAI_ENDPOINT"] = input("Enter your Azure OpenAI Endpoint (e.g., https://your-resource.openai.azure.com/): ").strip()
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = input("Enter your GPT deployment name: ").strip()
            os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] = input("Enter your embedding deployment name: ").strip()
            api_version = input("Enter API version (default: 2024-02-15-preview): ").strip()
            os.environ["AZURE_OPENAI_API_VERSION"] = api_version or "2024-02-15-preview"
        else:
            api_key = getpass.getpass("Enter your OpenAI API Key: ")
            os.environ["OPENAI_API_KEY"] = api_key
    
    documents_dir = "./documents"
    if len(sys.argv) > 1:
        documents_dir = sys.argv[1]
    
    agent = setup_knowledge_base(documents_dir)
    
    if agent is None:
        print("\nFailed to set up knowledge base. Exiting.")
        return

    chat_loop(agent)


if __name__ == "__main__":
    main()