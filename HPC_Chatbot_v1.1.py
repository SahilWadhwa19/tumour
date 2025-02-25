import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

class Pipeline:
    
    class Valves(BaseModel):
        MODEL_NAME: str
        
    def __init__(self):
        self.llm = None
        self.valves = self.Valves(
            **{
                "MODEL_NAME": os.getenv("MODEL_NAME", "deepseek-r1:8b"),
            }
        )
        
    async def on_startup(self):
        os.environ["GROQ_API_KEY"] = "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"
        global llm
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
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
        
        response = self.llm.invoke(user_message)
        
        return response.content
