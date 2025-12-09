from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr
import os

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY environment variable is not set")
llm = ChatGroq(api_key = SecretStr(groq_api_key), model = 'llama-3.3-70b-versatile')

if __name__ == '__main__':
    response = llm.invoke("Why is the sky blue?")
    print(response.content)