#!/usr/bin/env python3
"""
Vizor Vector Memory
ChromaDB-based vector memory for knowledge storage and retrieval
"""

import chromadb
from chromadb.config import Settings
import asyncio
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import json
import time
from pathlib import Path

class VectorMemory:
    """
    Vector-based memory system using ChromaDB
    
    Stores and retrieves knowledge using semantic similarity
    Integrates with the reasoning engine for context-aware responses
    """
    
    def __init__(self, config):
        self.config = config
        self.db_path = config.data_dir / "chroma_db"
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            config.vector_config.embedding_model
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get or create the main knowledge collection"""
        try:
            collection = self.client.get_collection(
                name=self.config.vector_config.collection_name
            )
        except:
            collection = self.client.create_collection(
                name=self.config.vector_config.collection_name,
                metadata={"description": "Vizor knowledge base"}
            )
        
        return collection
    
    async def add_document(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the vector memory
        
        Args:
            content: Document content
            metadata: Optional metadata
            doc_id: Optional document ID (auto-generated if None)
            
        Returns:
            Document ID
        """
        if not doc_id:
            doc_id = f"doc_{int(time.time() * 1000)}"
        
        if not metadata:
            metadata = {}
        
        # Add timestamp
        metadata['timestamp'] = time.time()
        metadata['content_length'] = len(content)
        
        # Generate embedding
        embedding = await asyncio.to_thread(
            self.embedding_model.encode, 
            content
        )
        
        # Add to collection
        self.collection.add(
            documents=[content],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_domain: Optional[str] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_domain: Optional domain filter
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        query_embedding = await asyncio.to_thread(
            self.embedding_model.encode, 
            query
        )
        
        # Build where clause for filtering
        where_clause = {}
        if filter_domain:
            where_clause['domain'] = filter_domain
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause if where_clause else None
        )
        
        # Format results
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
                
                # Apply similarity threshold
                if min_similarity and similarity < min_similarity:
                    continue
                
                documents.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity,
                    'id': results['ids'][0][i]
                })
        
        return documents
    
    async def add_threat_intelligence(
        self, 
        intel_data: Dict[str, Any],
        source: str
    ) -> str:
        """Add threat intelligence to memory"""
        
        content = f"""
        Threat Intelligence from {source}:
        
        Title: {intel_data.get('title', 'Unknown')}
        Description: {intel_data.get('description', '')}
        Indicators: {', '.join(intel_data.get('indicators', []))}
        Threat Actors: {', '.join(intel_data.get('threat_actors', []))}
        TTPs: {', '.join(intel_data.get('ttps', []))}
        Severity: {intel_data.get('severity', 'Unknown')}
        """.strip()
        
        metadata = {
            'type': 'threat_intelligence',
            'source': source,
            'domain': 'threat_intel',
            'severity': intel_data.get('severity', 'unknown'),
            'indicators_count': len(intel_data.get('indicators', [])),
            'threat_actors': intel_data.get('threat_actors', [])
        }
        
        return await self.add_document(content, metadata)
    
    async def add_vulnerability_data(
        self, 
        vuln_data: Dict[str, Any]
    ) -> str:
        """Add vulnerability data to memory"""
        
        content = f"""
        Vulnerability Information:
        
        CVE: {vuln_data.get('cve', 'Unknown')}
        CVSS Score: {vuln_data.get('cvss_score', 'Unknown')}
        Description: {vuln_data.get('description', '')}
        Affected Products: {', '.join(vuln_data.get('affected_products', []))}
        Mitigation: {vuln_data.get('mitigation', '')}
        References: {', '.join(vuln_data.get('references', []))}
        """.strip()
        
        metadata = {
            'type': 'vulnerability',
            'cve': vuln_data.get('cve'),
            'domain': 'vulnerabilities',
            'cvss_score': vuln_data.get('cvss_score', 0),
            'severity': vuln_data.get('severity', 'unknown'),
            'affected_products': vuln_data.get('affected_products', [])
        }
        
        return await self.add_document(content, metadata)
    
    async def add_learning_content(
        self, 
        topic: str, 
        content: str, 
        source: str
    ) -> str:
        """Add learning content to memory"""
        
        formatted_content = f"""
        Learning Content: {topic}
        Source: {source}
        
        {content}
        """.strip()
        
        metadata = {
            'type': 'learning_content',
            'topic': topic,
            'source': source,
            'domain': 'learning',
            'content_type': 'educational'
        }
        
        return await self.add_document(formatted_content, metadata)
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        
        try:
            # Get collection count
            collection_count = self.collection.count()
            
            # Get documents by type
            all_docs = self.collection.get()
            
            type_counts = {}
            domain_counts = {}
            
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    doc_type = metadata.get('type', 'unknown')
                    domain = metadata.get('domain', 'unknown')
                    
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            return {
                'total_documents': collection_count,
                'types': type_counts,
                'domains': domain_counts,
                'collection_name': self.config.vector_config.collection_name,
                'embedding_model': self.config.vector_config.embedding_model
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'total_documents': 0,
                'types': {},
                'domains': {}
            }
    
    async def cleanup_old_documents(self, days: int = 90):
        """Clean up old documents based on retention policy"""
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            ids_to_delete = []
            
            if all_docs['metadatas']:
                for i, metadata in enumerate(all_docs['metadatas']):
                    doc_timestamp = metadata.get('timestamp', 0)
                    if doc_timestamp < cutoff_time:
                        ids_to_delete.append(all_docs['ids'][i])
            
            # Delete old documents
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                
            return {
                'deleted_count': len(ids_to_delete),
                'cutoff_date': cutoff_time
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'deleted_count': 0
            }
    
    async def export_memory(self, export_path: str) -> Dict[str, Any]:
        """Export memory to JSON file"""
        
        try:
            all_docs = self.collection.get()
            
            export_data = {
                'collection_name': self.config.vector_config.collection_name,
                'export_timestamp': time.time(),
                'document_count': len(all_docs['ids']) if all_docs['ids'] else 0,
                'documents': []
            }
            
            if all_docs['ids']:
                for i in range(len(all_docs['ids'])):
                    doc_data = {
                        'id': all_docs['ids'][i],
                        'content': all_docs['documents'][i],
                        'metadata': all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                    }
                    export_data['documents'].append(doc_data)
            
            # Save to file
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return {
                'success': True,
                'export_path': str(export_file),
                'document_count': export_data['document_count']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def import_memory(self, import_path: str) -> Dict[str, Any]:
        """Import memory from JSON file"""
        
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                return {
                    'success': False,
                    'error': f"Import file not found: {import_path}"
                }
            
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            for doc in import_data.get('documents', []):
                await self.add_document(
                    content=doc['content'],
                    metadata=doc.get('metadata', {}),
                    doc_id=doc['id']
                )
                imported_count += 1
            
            return {
                'success': True,
                'imported_count': imported_count,
                'total_available': len(import_data.get('documents', []))
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
