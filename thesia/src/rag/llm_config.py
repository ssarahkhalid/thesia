from langchain_openai import OpenAI  # Updated import
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def create_llm(api_key, model_name="gpt-3.5-turbo-instruct"):  # Updated model name
    callbacks = [StreamingStdOutCallbackHandler()]  # Updated to use callbacks
    
    # Use the api_key parameter to initialize the OpenAI object
    llm = OpenAI(
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=2000,
        top_p=1,
        callbacks=callbacks,  # Updated to use callbacks
        verbose=True,
    )
    return llm