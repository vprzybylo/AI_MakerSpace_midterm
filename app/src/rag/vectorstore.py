import logging
import os

from langchain_community.vectorstores import Qdrant

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.collection_name = "grid_code"

    def create_vectorstore(self, documents):
        """Create vector store."""
        logger.info("Creating vector store...")
        vectorstore = Qdrant.from_documents(
            documents=documents,
            embedding=self.embedding_model.model,
            location=":memory:",  # Use in-memory storage
            collection_name=self.collection_name,
        )
        logger.info(f"Created vector store with {len(documents)} chunks")
        return vectorstore

    def similarity_search(self, query, k=4):
        raise NotImplementedError("Use the Qdrant vectorstore instance directly")
