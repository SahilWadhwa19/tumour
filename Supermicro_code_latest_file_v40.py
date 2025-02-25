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
        llm = ChatOllama(
            model=self.valves.MODEL_NAME,
            temperature=0.7,  # Make sure to add a comma here
            base_url= "http://172.28.6.4:11434",
        )
        response = llm.invoke(user_message)
        
        return user_message
