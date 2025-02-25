from langchain_ollama import ChatOllama
class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_MODEL_NAME: str
        
    def __init__(self):
        self.valves = self.Valves(
            **{
                "MODEL_NAME": os.getenv("MODEL_NAME", "deepseek-r1:8b"),
            }
        )

    async def on_startup(self):
        from langchain_ollama import ChatOllama
        
        global llm
        self.llm = ChatOllama(
        model=self.valves.MODEL_NAME
        temperature=0.7,
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
        response = self.llm.invoke("What is great in AI field in a nice comprehensive style")
        
        return response.content