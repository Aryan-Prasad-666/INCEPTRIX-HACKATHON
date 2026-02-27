from supermemory import Supermemory
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

supermemory_key = os.getenv('ARYAN_SUPERMEMORY_API_KEY')
gemini_key = os.getenv('ARYAN_GEMINI_KEY')
groq_key = os.getenv('groq_api_key')

client = Supermemory(api_key=supermemory_key)

from openai import OpenAI

groq_client = OpenAI(
    api_key=groq_key,
    base_url="https://api.supermemory.ai/v3/https://api.groq.com/openai/v1",
    default_headers={
        "x-supermemory-api-key": supermemory_key,
        "x-sm-user-id": "user_123"
    }
)

def chat_with_groq():
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "user", "content": "what do you know about me?"}
            ],
            max_tokens=1000
        )

        print("Groq Response:", response.choices[0].message.content)
        return response
    except Exception as error:
        print(f"Error with Groq: {error}")

chat_with_groq()