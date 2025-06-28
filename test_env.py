import os
from dotenv import load_dotenv

load_dotenv()  # This loads .env into the environment

api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", api_key if api_key else "Key not found")
