import logging
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = "stock_data"

FINANCIAL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a financial data analyst assistant with access to real market data.
Use ONLY the context below to answer the question. Be specific with numbers and metrics.
If the data does not contain enough information, say so clearly.

Context from financial database:
{context}

Question: {question}

Answer (be concise, use specific numbers from the data):""",
)


class FinancialRAGChain:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2'
        )
        self.llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.vectorstore = None
        self.chain = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        count = self.vectorstore._collection.count()
        logger.info(f"ChromaDB loaded: {count} documents")

    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": FINANCIAL_QA_PROMPT},
            return_source_documents=True,
        )
        logger.info("RAG chain ready")

    def query(self, question: str) -> dict:
        if self.chain is None:
            self._build_chain()
        try:
            result = self.chain.invoke({"query": question})
            sources = [
                doc.metadata.get("ticker", "unknown")
                for doc in result.get("source_documents", [])
            ]
            return {
                "answer": result["result"],
                "sources": list(set(sources)),
                "num_docs_retrieved": len(result.get("source_documents", [])),
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "num_docs_retrieved": 0,
            }


_rag_chain_instance = None


def get_rag_chain() -> FinancialRAGChain:
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = FinancialRAGChain()
    return _rag_chain_instance
