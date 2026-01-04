from typing import List, Dict
import os
from pypdf import PdfReader
from docx import Document


try:
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


class DocumentChunker:
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:  # Only break if we're past halfway
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = len(chunks)
            
            chunks.append({
                'text': chunk_text.strip(),
                'metadata': chunk_metadata
            })
            
            start = end - self.chunk_overlap
            
        return chunks
    
    def load_documents_from_directory(self, directory: str) -> List[Dict]:
        documents = []
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return documents
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            try:
                if filename.endswith('.pdf'):
                    if not HAS_PDF:
                        print(f"Skipping {filename}: pypdf not installed. Run: pip install pypdf")
                        continue
                    
                    reader = PdfReader(filepath)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                    
                    documents.append({
                        'text': content,
                        'metadata': {'source': filename, 'filepath': filepath}
                    })
                
                elif filename.endswith('.docx'):
                    if not HAS_DOCX:
                        print(f"Skipping {filename}: python-docx not installed. Run: pip install python-docx")
                        continue
                    
                    doc = Document(filepath)
                    content = "\n".join([para.text for para in doc.paragraphs])
                    
                    documents.append({
                        'text': content,
                        'metadata': {'source': filename, 'filepath': filepath}
                    })
                
                elif filename.endswith('.txt') or filename.endswith('.md'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({
                            'text': content,
                            'metadata': {'source': filename, 'filepath': filepath}
                        })
                        
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return documents
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['text'], doc.get('metadata', {}))
            all_chunks.extend(chunks)
        
        return all_chunks
