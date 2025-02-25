import os
from typing import List, Union, Generator, Iterator
from langchain_ollama import ChatOllama
from pydantic import BaseModel

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
        # Initialize ChatOllama model with the given parameters
        self.llm = ChatOllama(
            model=self.valves.MODEL_NAME,
            temperature=0.7,  # Make sure to add a comma here
        )

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
        
        # Use self.llm.invoke() if this method works as expected for the library
        response = self.llm.invoke("What is great in AI field in a nice comprehensive style")
        
        return response.content