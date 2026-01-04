import json
from typing import List, Dict, Optional
from vector_store import VectorStore


class RetrievalTool:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> str:
        results = self.vector_store.retrieve(query, top_k=top_k)
        
        if not results:
            return "No relevant documents found."
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            source = doc['metadata'].get('source', 'Unknown')
            text = doc['text']
            formatted_results.append(
                f"[Document {i} - Source: {source}]\n{text}\n"
            )
        
        return "\n".join(formatted_results)
    
    def get_tool_definition(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "retrieve_knowledge",
                "description": "Retrieve relevant documents from the knowledge base to answer questions. Use this when you need domain-specific information to answer a question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant documents"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of documents to retrieve (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute_tool_call(self, tool_call) -> str:
        args = json.loads(tool_call.function.arguments)
        query = args.get('query', '')
        top_k = args.get('top_k', 5)
        return self.retrieve(query, top_k)
