from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class GridCodePDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def load_and_split(self):
        """Load PDF and split into chunks"""
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        return self.text_splitter.split_documents(pages)
    
    def extract_metadata(self):
        """Extract metadata from PDF like sections, tables etc."""
        # TODO: Implement metadata extraction
        pass 