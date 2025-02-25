import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_ollama import ChatOllama

class Pipeline:
    
    class Valves(BaseModel):
        MODEL_NAME: str
        
    def __init__(self):
        self.valves = self.Valves(
            **{
                "MODEL_NAME": os.getenv("MODEL_NAME", "deepseek-r1:8b"),
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
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        response = llm.invoke(user_message)
        
        return response.content
