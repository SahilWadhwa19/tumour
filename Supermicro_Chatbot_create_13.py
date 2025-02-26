import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class Pipeline:
    
    class Valves(BaseModel):
        MODEL_NAME: str
        
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.sample_data = None
        self.valves = self.Valves(
            **{
                "MODEL_NAME": os.getenv("MODEL_NAME", "llama3-70b-8192"),
            }
        )
        
    async def on_startup(self):
        os.environ["GOOGLE_API_KEY"]="AIzaSyDf5jdwzdhEpjip3aEB0sywg9htgYy3RUA"
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.sample_data = "All will be great"
        pass
        
        
    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)
        
        cwd = os.getcwd()
        
        # Print the current working directory
        print("Current working directory:", cwd)
        return self.sample_data
