"""
RAG Engine for Bella Roma AI Restaurant Bot.
Loads compressed menu data, builds a FAISS vector store,
and answers menu-related queries using Groq LLM and HuggingFace embeddings.
"""

import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Path to compressed menu data
DATA_DIR = Path(__file__).parent / "data"
COMPRESSED_MENU_PATH = DATA_DIR / "compressed_menu.json"


class RAGEngine:
    """Retrieval-Augmented Generation engine for menu queries."""

    def __init__(self):
        """Initialize the RAG engine with embeddings, vector store, and QA chain."""
        self.documents = self._load_documents()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = self._build_vector_store()
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        self.qa_chain = self._build_qa_chain()

    def _load_documents(self) -> list[Document]:
        """Load compressed menu chunks and convert to LangChain Documents."""
        with open(COMPRESSED_MENU_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        documents = [
            Document(page_content=chunk, metadata={"source": "menu"})
            for chunk in chunks
        ]
        return documents

    def _build_vector_store(self) -> FAISS:
        """Create a FAISS vector store from the loaded documents."""
        vector_store = FAISS.from_documents(self.documents, self.embeddings)
        return vector_store

    def _build_qa_chain(self):
        """Build the RAG chain with a strict system prompt using LCEL."""
        system_template = """You are Bella Roma's friendly restaurant assistant. 
You ONLY answer questions using the provided context about the restaurant's menu.
If the answer cannot be found in the context, respond with:
"I'm sorry, that item is not on our menu."

Do NOT make up information. Do NOT reference items not in the context.
Be warm, helpful, and concise in your responses.
Use a friendly Italian-restaurant tone.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate.from_template(system_template)

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return qa_chain

    def query(self, question: str) -> str:
        """
        Answer a menu-related question using RAG.

        Args:
            question: The user's question about the menu.

        Returns:
            A string response based on the retrieved context.
        """
        try:
            result = self.qa_chain.invoke(question)
            return result
        except Exception as e:
            return f"I'm sorry, something went wrong while looking that up. Please try again."
