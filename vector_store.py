from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI, AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    #Manages document embeddings and retrieval using ChromaDB.
    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # if os.getenv("AZURE_OPENAI_API_KEY"):
        try:
            self.openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
            print(f"✓ Using Azure OpenAI")
            print(f"  Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
            print(f"  Embedding Model: {self.embedding_model}")
            print(f"  API Version: {os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')}")
        except Exception as e:
            print(f"✗ Error initializing Azure OpenAI: {e}")
            raise
        # else:
        #     self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #     self.embedding_model = "text-embedding-3-small"
    
    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"✗ Error generating embedding: {e}")
            print(f"  Model: {self.embedding_model}")
            print(f"  Text length: {len(text)}")
            raise
    
    def add_documents(self, chunks: List[Dict]) -> None:
        if not chunks:
            print("No chunks to add")
            return
        
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(chunks)} chunks to vector store")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            query_embedding = self.get_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    retrieved_docs.append({
                        'text': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results.get('distances') else None
                    })
            
            print(f"✓ Retrieved {len(retrieved_docs)} documents for query: '{query[:50]}...'")
            return retrieved_docs
        except Exception as e:
            print(f"✗ Error during retrieval: {e}")
            raise
    
    def clear_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared")
    
    def count_documents(self) -> int:
        return self.collection.count()
