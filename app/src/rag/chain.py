from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model="gpt-4o")
        self.chain = self._create_chain()
    
    def _create_chain(self):
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for field workers in the electricity transmission sector.
        Answer questions about the Grid Code using the following context.
        If you're unsure or the context doesn't contain the answer, say so.
        
        Context: {context}
        Question: {input}
        """)
        
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(
            self.vectorstore.as_retriever(), 
            document_chain
        )
        return retrieval_chain
    
    def invoke(self, question):
        return self.chain.invoke({"input": question}) 