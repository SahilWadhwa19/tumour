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
        self.prompt = None
        self.database = None
        self.chain = None
        self.embeddings = None
        self.retriever = None
        self.retriever_chain = None
        self.valves = self.Valves(
            **{
                "MODEL_NAME": os.getenv("MODEL_NAME", "llama3-70b-8192"),
            }
        )
        
    async def on_startup(self):
        
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
        os.environ["GROQ_API_KEY"] = "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"
        llm = ChatGroq(
            model=self.valves.MODEL_NAME,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        
        
        os.environ["GOOGLE_API_KEY"]="AIzaSyDf5jdwzdhEpjip3aEB0sywg9htgYy3RUA"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        database=FAISS.load_local(
        "/app/faiss_index_latest_db_6", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True
        )

        prompt = ChatPromptTemplate.from_template("""
        You are an experienced HPC and Datacenter Solutions Presales Engineer. You provide insights and assistance to other engineers and sales persons to enable them to find appropriate products and solutions from our portfolio of products and roadmaps provided in the augmented data set. 
        
        You should try to be as accurate as possible, but provide potential solutions if you are unable to find sufficiant data, but explain if suggestions may require further confirmation and development if presented.
        
        <context>
        {context}
        </context>
        Question: {input}""")
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=database.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        # response = retrieval_chain.invoke({"input":user_message})
        response = llm.invoke(user_message)
        return response.content
