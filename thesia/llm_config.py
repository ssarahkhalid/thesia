from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def create_llm(model_path="models/llama-2-7b-chat.gguf"):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,
        n_ctx=4096,
    )
    return llm 