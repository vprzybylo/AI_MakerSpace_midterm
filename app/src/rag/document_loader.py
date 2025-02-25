from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import logging

logger = logging.getLogger(__name__)

class GridCodeLoader:
    def __init__(self, file_path, pages=None):
        self.file_path = file_path
        self.pages = pages
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def load_and_split(self):
        logger.info(f"Loading PDF from {self.file_path}")
        # Open PDF directly first to get total pages
        reader = pypdf.PdfReader(self.file_path)
        total_pages = len(reader.pages)
        
        if isinstance(self.pages, int):
            # Load first n pages
            pages_to_load = list(range(min(self.pages, total_pages)))
            logger.info(f"Loaded first {len(pages_to_load)} pages from PDF")
        elif isinstance(self.pages, (list, tuple)):
            # Load specific pages
            pages_to_load = [p for p in self.pages if p < total_pages]
            logger.info(f"Loaded pages {self.pages} from PDF")
        else:
            pages_to_load = list(range(total_pages))
            logger.info(f"Loaded all {len(pages_to_load)} pages from PDF")
        
        # Now use PyPDFLoader with the selected pages
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        documents = [doc for i, doc in enumerate(documents) if i in pages_to_load]
        
        logger.info("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks 