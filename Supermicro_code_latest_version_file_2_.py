import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

class Pipeline:

    class Valves(BaseModel):
        MODEL_NAME: str

    def __init__(self):
        pass

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
        
        
        return user_message
